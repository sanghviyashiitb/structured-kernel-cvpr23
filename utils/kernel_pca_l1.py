import time
import numpy as np
from numpy.linalg import norm
from numpy.fft import fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import savemat, loadmat
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.blind_deconv import Normalize_Kernel, shrinkage_torch, coarse_kernel_estimation
from utils.blind_deconv import multiscale_kernel1, multiscale_kernel2
from utils.utils_deblur import gauss_kernel, pad, crop, psf2otf, otf2psf, D, imresize, shock
from utils.utils_torch import conv_fft_batch, psf_to_otf, p4ip_denoiser
from utils.utils_torch import tens_to_img, scalar_to_tens, img_to_tens
from utils.dataloader import center_kernel
from utils.gauss_kernel_estimate import estimate_gaussian_kernel 

from models.network_p4ip import P4IP_Net
from models.ResUNet import ResUNet
from models.deep_weiner.deblur import DEBLUR

global device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


np.random.seed(34)
def nan(t):
	return torch.isnan(t).any().item()

def L2_LOSS_GRADIENT(x, y):
	Dx_x, Dy_x = torch.gradient(x, dim=[2,3])
	Dx_y, Dy_y = torch.gradient(y, dim=[2,3])
	L2_LOSS = nn.MSELoss()
	return L2_LOSS(Dx_x, Dx_y) + L2_LOSS(Dy_x, Dy_y)

# def kernel_to_z(kernel, V, mean):
# 	_, N, K = V.size()
# 	return torch.matmul(V.transpose(1,2), kernel.float().view(1,N,1) - mean)

def z_to_kernel(z, V, mean, K_N):
	_, N, K = V.size()
	k_vec =  torch.matmul(V, z) + mean
	k_vec = torch.clamp(k_vec,0,np.inf)
	k_vec = torch.div(k_vec, torch.sum(k_vec))

	return k_vec.view(1,1,K_N,K_N)

"""
-----------------------------------------------------------------------------------------

Iterative Scheme for Blind Deconvolution starts here
-----------------------------------------------------------------------------------------
"""
def iterative_scheme(y, M, deconv, opts = {} ):
	# 'circular' or 'symmetric' convolution assumption, choose latter for real experiments
	MODE = opts['MODE'] if 'MODE' in opts else 'symmetric' 
	# Kernel search space size
	K_N = opts['K_N'] if 'K_N' in opts else 32
	# Use kernel-estimation module for initialization
	USE_KERNEL_EST = opts['USE_KERNEL_EST'] if 'USE_KERNEL_EST' in opts else False
	# L1-prior hyperparameters
	MU = opts['MU'] if 'MU' in opts else 1e-5
	RHO = opts['RHO'] if 'RHO' in opts else 1e-5
	# Optimization hyperparameters
	TOL = opts['TOL'] if 'TOL' in opts else 1e-3
	MAX_ITERS = opts['MAX_ITERS'] if 'MAX_ITERS' in opts else 100
	STEP_SIZE = opts['STEP_SIZE'] if 'STEP_SIZE' in opts else 0.1
	# Display the kernel while running the scheme
	SHOW_KERNEL = opts['SHOW_KERNEL'] if 'SHOW_KERNEL' in opts else False
	# Print out value of cost function, step size and other relevant parameters during iterations
	VERBOSE = opts['VERBOSE'] if 'VERBOSE' in opts else False

	L2_LOSS = nn.MSELoss()
	NORMALIZER = Normalize_Kernel(); NORMALIZER.to(device)
	# Load Kernel-PCA components
	struct = loadmat('data/kernel_pca.mat'); V, mean, d = struct['V'], struct['mean'], struct['D']
	_, K_pca = np.shape(V)
	# Symmetric mode requires the image to be padded and then the relevant output be cropped out
	# after deconvolution
	H, W = np.shape(y)
	H1, W1 = H//4, W//4
	if MODE == 'symmetric':
		y_pad = np.pad(y, ((H1,H1),(W1,W1)), mode='symmetric')
	# Get the denoised but blurred image
	yn =  p4ip_denoiser(y, M, deconv.denoiser)	
	yn_pad =  p4ip_denoiser(y_pad, M, deconv.denoiser)
	# Get Pytorch version of some vectors
	yt = img_to_tens(y).to(device)
	yt_pad = img_to_tens(y_pad).to(device)
	yn_t = img_to_tens(yn_pad).to(device)
	Mt = scalar_to_tens(M).to(device)
	
	# Get an initial estimate of kernel
	if USE_KERNEL_EST:
		# Using a trained kernel estimation module
		with torch.no_grad():
			k0_torch = deconv.est_kernel(yt, Mt)
		k0 = pad(center_kernel(tens_to_img(k0_torch, [32,32])), [K_N, K_N]) 
	else:
		# Using a traditional alternating minimization kernel estimation
		# METHOD-I
		# _, k0 = multiscale_kernel2(yn, [K_N])	
		# METHOD-II
		# _, k0 = multiscale_kernel1(y, M, deconv.non_blind_solver, deconv.denoiser, [K_N])	
		# METHOD-III
		k0, _ = estimate_gaussian_kernel(yn, k_size = K_N, C = 99/255.0)
	z0 = np.matmul(V.T, np.reshape(k0, [K_N**2,1])-mean)
	
	# Convert latent estimate to PyTorch vector
	z = img_to_tens(z0).to(device).float()
	z1 = img_to_tens(z0).to(device).float() # Proxy variable for HQS 
	Vt = img_to_tens(V, [1,K_N**2, K_pca]).float().to(device)
	mean_t = img_to_tens(mean,[1,K_N**2,1]).float().to(device)
	weights = img_to_tens(1/(np.sqrt(d)+1e-6), z.size()).float().to(device)


	del_z = torch.zeros(z.size()).to(device)	
	z.requires_grad = True; del_z.requires_grad = False
	
	loss_prev = np.inf
	x_list, k_list = [],  []
	# Iterative scheme starts here
	iterations = 0 
	while iterations < MAX_ITERS:
		deconv.non_blind_solver.zero_grad()
		k = z_to_kernel(z, Vt, mean_t, K_N)
		_, A = psf_to_otf(k, yt_pad.size()); A = A.to(device)
		# Send to non-blind solver
		x_rec_list = deconv.non_blind_solver(yt_pad, k, Mt)
		x_rec = x_rec_list[-1]
		# Get current kernel estimate in FFT-Form
		# Reconstruct the blurred image
		y_rec = conv_fft_batch(A, x_rec)
		# MSE with denoised-only image as target
		loss = L2_LOSS(yn_t, y_rec) 
		# This step calculates the gradient of z w.r.t to the loss function
		loss.backward() 
		with torch.no_grad():
			# Gradient descent step
			del_z = z.grad
			del_z += MU*(z-z1)
			z = z.sub_(STEP_SIZE*del_z)
			# Update z1 using shrinkage
			z1 = shrinkage_torch(z, RHO*weights)

		del_loss = (loss_prev - loss.item())/loss_prev 
		loss_prev = loss.item()

		# Reduce step size if the cost function is not decreasing 
		if del_loss < 0:
			STEP_SIZE *= 0.5
		# and terminate the scheme if the scheme has converged	
		if STEP_SIZE < TOL:
			break
		MU *= 1.01
		RHO /= 1.001		# Print out values if needed
		if VERBOSE:
			print('iterations: %d, loss fn: %0.6f, current step size: %0.3f'%(iterations,loss.item()*1e3, STEP_SIZE))
		# Book-keeping: Tracking loss, current kernel estimate and image
		k_np, x_np = tens_to_img(k, [K_N,K_N]), tens_to_img(x_rec)
		if MODE == 'symmetric':
			x_np = x_np[H1:H1+H, W1:W1+W]
		k_list.append(k_np); x_list.append(x_np) 
		iterations += 1
		# FOR THE REMOTE POSSIBILITY THAT THE THING CONVERGES TO NAN
		# RESTART THE SCHEME WITH A VERY SMALL PERTUBATION TO y
		if nan(k):
			with torch.no_grad():
				y = np.clip(y + 1e-4*np.random.normal(0,1,np.shape(y)), 0, np.inf)
				yt = img_to_tens(y.astype(np.float32)).to(device)
				# k, k1 - HQS variables for f(.) and g(.) respectively
				z = img_to_tens(z0).to(device).float()
				z1 = z.detach().clone().to(device) 
				del_z = torch.zeros(z.size()).to(device)
				z.requires_grad = True;
				iterations = 0
				loss_prev = np.inf
				if VERBOSE:
					print('restarting scheme')

	return x_list, k_list