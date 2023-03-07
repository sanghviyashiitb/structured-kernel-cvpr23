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
from utils.utils_torch import conv_fft_batch, psf_to_otf, p4ip_denoiser,net_wrapper, p4ip_wrapper
from utils.utils_torch import tens_to_img, scalar_to_tens, img_to_tens
from utils.dataloader import center_kernel
from utils.gauss_kernel_estimate import estimate_gaussian_kernel 

from models.network_p4ip import P4IP_Net
from models.ResUNet import ResUNet
from models.deep_weiner.deblur import DEBLUR

global device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def nan(t):
	return torch.isnan(t).any().item()

def L2_LOSS_GRADIENT(x, y):
	_, _, H, W = x.size()
	Dx_x, Dy_x = torch.gradient(x.view(H,W), dim=[0,1])
	Dx_y, Dy_y = torch.gradient(y.view(H,W), dim=[0,1])
	L2_LOSS = nn.MSELoss()
	return L2_LOSS(Dx_x, Dx_y) + L2_LOSS(Dy_x, Dy_y)

def L1_LOSS_GRADIENT(x, y):
	_, _, H, W = x.size()
	Dx_x, Dy_x = torch.gradient(x.view(H,W), dim=[0,1])
	Dx_y, Dy_y = torch.gradient(y.view(H,W), dim=[0,1])
	L1_LOSS = nn.L1Loss()
	return L1_LOSS(Dx_x, Dx_y) + L1_LOSS(Dy_x, Dy_y)

def get_initial_z(sigma0, theta, kernel_mlp):
	N = kernel_mlp.n_control_points()
	sigma1 = np.clip(sigma0*4, 0, 32)/(N-1)
	xx, yy = sigma1*np.cos(theta*np.pi/180), sigma1*np.sin(theta*np.pi/180)
	z = np.zeros([2*(N-1)], dtype=np.float32)
	for idx in range(N-1):
		z[2*idx] = (idx+1)*xx
		z[2*idx+1] = (idx+1)*yy		
	zt = img_to_tens(z).to(device)
	with torch.no_grad():
		kt = kernel_mlp(zt)	
	return kt, zt

"""
-----------------------------------------------------------------------------------------

Iterative Scheme for Blind Deconvolution starts here
-----------------------------------------------------------------------------------------
"""


def iterative_scheme(y, M, networks, opts = {} ):
	"""
	Component networks: non-blind solver, denoiser, kernel-network
	"""
	p4ip = networks['nb_solver']
	denoiser = networks['denoiser']
	kernel_mlp = networks['kernel_mlp']
	
	"""
	Iterative scheme options
	"""	
	# 'circular' or 'symmetric' convolution assumption, choose latter for real-blur
	MODE = opts['MODE'] if 'MODE' in opts else 'symmetric' 
	# Use kernel-estimation module for initialization
	USE_KERNEL_EST = opts['USE_KERNEL_EST'] if 'USE_KERNEL_EST' in opts else False
	# Optimization hyperparameters
	TOL = opts['TOL'] if 'TOL' in opts else 1e-8
	RHO = opts['RHO'] if 'RHO' in opts else 1e-4
	MAX_ITERS = opts['MAX_ITERS'] if 'MAX_ITERS' in opts else 150
	STEP_SIZE = opts['STEP_SIZE'] if 'STEP_SIZE' in opts else 1e5
	STEP_SIZE2 = opts['STEP_SIZE2'] if 'STEP_SIZE2' in opts else 2.0
	# Print out value of cost function, step size and other relevant parameters during iterations
	VERBOSE = opts['VERBOSE'] if 'VERBOSE' in opts else False
	# Use gradient loss
	USE_GRADIENT_LOSS = opts['USE_GRADIENT_LOSS'] if 'USE_GRADIENT_LOSS' in opts else False
	SECOND_STAGE = opts['SECOND_STAGE'] if 'SECOND_STAGE' in opts else True
	FIRST_STAGE = opts['FIRST_STAGE'] if 'FIRST_STAGE' in opts else True

	if USE_GRADIENT_LOSS:
		# L2_LOSS = L1_LOSS_GRADIENT
		L2_LOSS = L2_LOSS_GRADIENT 
	else:
		L2_LOSS = nn.MSELoss()
	K_N = 64
	# Symmetric mode requires the image to be padded and then the relevant output be cropped out
	# after deconvolution
	H, W = np.shape(y)
	H1, W1 = H//4, W//4
	if MODE == 'symmetric':
		y_pad = np.pad(y, ((H1,H1),(W1,W1)), mode='symmetric')
	# Get the denoised but blurred image
	yn =  p4ip_denoiser(y, M, denoiser)	
	yn_pad =  p4ip_denoiser(y_pad, M, denoiser)
	
	# Get Pytorch version of some vectors
	yt = img_to_tens(y).to(device)
	yt_pad = img_to_tens(y_pad).to(device)
	yn_t = img_to_tens(yn).to(device)
	Mt = scalar_to_tens(M).to(device)

	# Get an initial estimate of kernel
	k_gauss, params = estimate_gaussian_kernel(yn, C = 101/255.0)
	sigma0, theta = params[0], params[2]
	kt, z0 = get_initial_z(sigma0, theta, kernel_mlp)
	k_np = tens_to_img(kt); k_np = k_np/np.sum(np.ravel(k_np))

	x_list, k_list, z_list = [],  [], []
	if FIRST_STAGE:
		# Convert kernel estimate to PyTorch vector
		z = z0.to(device)
		del_z = torch.zeros(z.size()).to(device)	
		z.requires_grad = True
		# Iterative scheme starts here
		iterations = 0
		loss_prev = np.inf; 
		while iterations < MAX_ITERS:
			p4ip.zero_grad(); kernel_mlp.zero_grad()
			k = kernel_mlp(z)
			_, A = psf_to_otf(k,  yt_pad.size()); A = A.to(device)
			# Send to non-blind solver
			x_rec_list = p4ip(yt_pad, k, Mt)
			x_rec = x_rec_list[-1]
			# Get current kernel estimate in FFT-Form
			# Reconstruct the blurred image
			y_rec = conv_fft_batch(A, x_rec)
			y_rec = y_rec[:,:,H1:H+H1,W1:W+W1]
			# MSE with denoised-only image as target
			loss = L2_LOSS(yn_t, y_rec) 
			# This step calculates the gradient of z w.r.t to the loss function
			loss.backward() 
			with torch.no_grad():
				z_list.append(tens_to_img(z))
				# Gradient descent step
				del_z = z.grad 
				z = z.sub_(STEP_SIZE*del_z)
				z.grad *= 0
				z.requires_grad = True
			del_loss = (loss_prev - loss.item())
			loss_prev = loss.item()
			if not nan(k):
				if VERBOSE:	
					print('iterations: %d, loss fn: %0.6f, current step size: %0.3f'%(iterations,loss.item()*1e3, STEP_SIZE))
				# Book-keeping: Tracking loss, current kernel estimate and image
				k_np, x_np = tens_to_img(k, [K_N,K_N]), tens_to_img(x_rec)
				if MODE == 'symmetric':
					x_np = x_np[H1:H1+H, W1:W1+W]
				k_list.append(k_np); x_list.append(np.clip(x_np,0,1)) 
				iterations += 1
				# Reduce step size if the cost function is not decreasing 
				# go back previous step
				if del_loss < 0 and np.abs(del_loss/loss.item()) > 1e-3:
					STEP_SIZE *= 0.5
					with torch.no_grad():
						z_np = z_list.pop(); z_np = z_list.pop()
						x_np = x_list.pop(); x_np = x_list.pop()
						z = img_to_tens(z_np).to(device)
						z.requires_grad = True
						iterations -= 1
				else:
					# and terminate the scheme if the scheme has converged
					if (del_loss<TOL and del_loss>0):
						break
			else:
				print("Breaking due to nan")
				break

	if SECOND_STAGE:
		NORMALIZER = Normalize_Kernel(); NORMALIZER.to(device)
		# Run a second stage with optimization in the kernel space instead of the latent space 
		k = img_to_tens(k_np).to(device); k.requires_grad = True
		k1 = img_to_tens(k_np).to(device) # Proxy variable for HQS 
		del_k = torch.zeros(k.size()).to(device)
		MU = 2.0; STEP_SIZE = STEP_SIZE2
		k0  = k_np
		# Iterative scheme starts here
		iterations = 0
		loss_prev = np.inf 
		while iterations < MAX_ITERS:
			p4ip.zero_grad(); 
			_, A = psf_to_otf(k,  yt_pad.size()); A = A.to(device)
			# Send to non-blind solver
			x_rec_list = p4ip(yt_pad, k, Mt)
			x_rec = x_rec_list[-1]
			# Get current kernel estimate in FFT-Form
			# Reconstruct the blurred image
			y_rec = conv_fft_batch(A, x_rec)
			y_rec = y_rec[:,:,H1:H+H1, W1:W+W1]
			# MSE with denoised-only image as target
			loss = L2_LOSS(yn_t, y_rec) 
			# This step calculates the gradient of z w.r.t to the loss function
			loss.backward() 
			with torch.no_grad():
				k_list.append(tens_to_img(k))
				# Gradient descent step
				del_k = k.grad + MU*(k-k1)
				k = k.sub_(STEP_SIZE*del_k)
				k = NORMALIZER(k)
				# Update k1 using shrinkage
				k1 = shrinkage_torch(k, RHO); k1 = NORMALIZER(k1); 
				k.requires_grad = True
			del_loss = (loss_prev - loss.item()) 
			loss_prev = loss.item()
			if not nan(k):
				# Book-keeping: Tracking loss, current kernel estimate and image
				k_np, x_np = tens_to_img(k, [K_N,K_N]), tens_to_img(x_rec)
				x_np = x_np[H1:H1+H, W1:W1+W]
				x_list.append(np.clip(x_np,0,1)) 
				iterations += 1
				if VERBOSE:	print('iterations: %d, loss fn: %0.6f, current step size: %0.3f'%(iterations,loss.item()*1e3, STEP_SIZE))
				# Reduce step size if the cost function is not decreasing and go back previos step
				if del_loss < 0 and np.abs(del_loss/loss.item()) > 1e-6:
					STEP_SIZE *= 0.5
					with torch.no_grad():
						k_np = k_list.pop(); k_np = k_list.pop()
						x_np = x_list.pop(); x_np = x_list.pop()
						k = img_to_tens(k_np).to(device)
						k.requires_grad = True
						iterations -= 1
				else:
					MU *= 1.01
					# and terminate the scheme if the scheme has converged
					if (del_loss<TOL and del_loss>0) or  STEP_SIZE < 1e-3:
						break
				
			# FOR THE REMOTE POSSIBILITY THAT THE THING CONVERGES TO NAN
			# RESTART THE SCHEME WITH A VERY SMALL PERTUBATION TO y
			else:
				print("Breaking due to nan")
				break
	return x_list, k_list