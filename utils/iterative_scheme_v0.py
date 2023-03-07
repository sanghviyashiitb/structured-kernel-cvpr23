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


"""
-----------------------------------------------------------------------------------------

Iterative Scheme for Blind Deconvolution starts here
-----------------------------------------------------------------------------------------
"""
def iterative_scheme(y, M, p4ip, denoiser, opts = {} ):
	# 'circular' or 'symmetric' convolution assumption, choose latter for real experiments
	MODE = opts['MODE'] if 'MODE' in opts else 'symmetric'
 	# Kernel search space size
	K_N = opts['K_N'] if 'K_N' in opts else 35
	# Use kernel-estimation module for initialization
	USE_KERNEL_EST = opts['USE_KERNEL_EST'] if 'USE_KERNEL_EST' in opts else False
	# L1-prior hyperparameters
	MU = opts['MU'] if 'MU' in opts else 2.0
	RHO = opts['RHO'] if 'RHO' in opts else 1e-4
	# Optimization hyperparameters
	TOL = opts['TOL'] if 'TOL' in opts else 1e-3
	MAX_ITERS = opts['MAX_ITERS'] if 'MAX_ITERS' in opts else 150
	STEP_SIZE = opts['STEP_SIZE'] if 'STEP_SIZE' in opts else 2
	UPDATE_RHO = opts['UPDATE_RHO'] if 'UPDATE_RHO' in opts else True
	# Display the kernel while running the scheme
	SHOW_KERNEL = opts['SHOW_KERNEL'] if 'SHOW_KERNEL' in opts else False
	# Print out value of cost function, step size and other relevant parameters during iterations
	VERBOSE = opts['VERBOSE'] if 'VERBOSE' in opts else False
	# Is the input to non-blind solver normalized?
	NORM_INPUT = opts['NORM_INPUT'] if 'NORM_INPUT' in opts else False
	DENOISE_TARGET = opts['DENOISE_TARGET'] if 'DENOISE_TARGET' in opts else True
	L2_LOSS = nn.MSELoss()
	NORMALIZER = Normalize_Kernel(); NORMALIZER.to(device)
	# Symmetric mode requires the image to be padded and then the relevant output be cropped out
	# after deconvolution
	H, W = np.shape(y)
	H1, W1 = H//4, W//4
	# Get the denoised but blurred image
	yn =  p4ip_denoiser(y, M, denoiser)	
	k0, _ = estimate_gaussian_kernel(yn, k_size=K_N, C = 99/255.0)
	
	if MODE == 'symmetric':
		y_pad = np.pad(y, ((H1,H1),(W1,W1)), mode='symmetric')
		yn_pad =  p4ip_denoiser(y_pad, M, denoiser)
		y_target = yn_pad if DENOISE_TARGET else y_pad/M
		# Get Pytorch version of some vectors
		yn_t = img_to_tens(y_target).to(device)
		yt = img_to_tens(y_pad).to(device)
	else:
		yn =  p4ip_denoiser(y, M, denoiser)
		y_target = yn if DENOISE_TARGET else y/M
		yt = img_to_tens(y).to(device)
		yn_t = img_to_tens(y_target).to(device)
		
	Mt = scalar_to_tens(M).to(device)
	if NORM_INPUT:
		yt_M = img_to_tens(y/M).to(device)
	# Get an initial estimate of kernel
	# Using a traditional alternating minimization kernel estimation
	# METHOD-I
	# _, k0 = multiscale_kernel2(yn, [K_N])	
	# METHOD-II
	# _, k0 = multiscale_kernel1(y, M, deconv.non_blind_solver, deconv.denoiser, [K_N])	
	# METHOD-III
	
	# Convert kernel estimate to PyTorch vector
	k = img_to_tens(k0).to(device) 
	k1 = img_to_tens(k0).to(device) 	# Proxy variable for HQS 
	del_k = torch.zeros([1,1,K_N,K_N]).to(device)	
	k.requires_grad = True; del_k.requires_grad = False
	
	loss_prev = np.inf
	x_list, k_list = [],  []; k_list.append(k0)
	# Iterative scheme starts here
	iterations = 0 
	while iterations < MAX_ITERS:
		# p4ip.zero_grad() 
		_, A = psf_to_otf(k, yt.size()); A = A.to(device)
		# Send to non-blind solver
		if NORM_INPUT:
			x_rec_list = p4ip(yt_M, k)
			x_rec = x_rec_list[-1]
		else:
			x_rec_list = p4ip(yt, k, Mt)
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
			del_k = k.grad
			del_k += MU*(k-k1)
			k = k.sub_(STEP_SIZE*del_k)
			# Normalize kernel entries 
			k = NORMALIZER(k); 
			k.requires_grad = True 
			# Update k1 using shrinkage
			k1 = shrinkage_torch(k, RHO); k1 = NORMALIZER(k1); 
		del_loss = (loss_prev - loss.item())/loss_prev 
		loss_prev = loss.item()

		# Reduce step size if the cost function is not decreasing 
		if del_loss < 0:
			STEP_SIZE *= 0.5
		# and terminate the scheme if the scheme has converged	
		if STEP_SIZE < TOL:
			break
		MU *= 1.01
		if UPDATE_RHO: RHO /= 1.001
		# Print out values if needed
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
				if MODE == 'symmetric':
					y_pad = np.pad(y, ((H1,H1),(W1,W1)), mode='symmetric')
					yt = img_to_tens(y_pad.astype(np.float32)).to(device)
				else:
					yt = img_to_tens(y.astype(np.float32)).to(device)
				# k, k1 - HQS variables for f(.) and g(.) respectively
				k = img_to_tens(k0).to(device) 
				k1 = k.detach().clone().to(device) 
				del_k = torch.zeros([1,1,K_N,K_N]).to(device)
				k.requires_grad = True; k1.requires_grad = False; del_k.requires_grad = False
				iterations = 0
				loss_prev = np.inf
				print('restarting scheme')
	return x_list, k_list



