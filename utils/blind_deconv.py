import numpy as np
from numpy.linalg import norm
from numpy.fft import fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from scipy.ndimage import sobel
import cv2 as cv

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils_torch import conv_fft, conv_fft_batch, psf_to_otf, img_to_tens, scalar_to_tens, p4ip_wrapper, unet_wrapper, p4ip_denoiser, sharpness
from utils.utils_deblur import shock, mask_gradients, psf2otf, otf2psf, imresize, D, Dt, shrinkage
from models.network_p4ip import P4IP_Net
from models.ResUNet import ResUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def multiscale_kernel2(y, k_sizes=[45]):
	
	H, W = np.shape(y)
	levels = len(k_sizes)
	y_sizes = []
	for idx in range(levels):
		factor = 2**(levels-1-idx)
		H1, W1 = int(H/factor), int(W/factor)
		y_sizes.append([H1,W1])

	for iters in range(levels):
		k_size = k_sizes[iters]; H1, W1 = y_sizes[iters]
		# Resize the current estimate of noisy and clean image
		y_i =  imresize(y, [H1, W1])
		if iters > 0: # Choose the blurred image as initial estimate for first iteration
			x_i = imresize(x_i, [H1, W1]) 
		else:
			x_i = y_i 

		# Run an iterative scheme which estimates the kernel and image alternatively
		# Kernel is estimated using l2-prior and image using TV-prior
		xb = shock(x_i, 5,1)
		for itrs in range(5):
			fx, fy = D(xb)
			gx, gy =  D(y_i)
			k_est = k_l2([fx, fy], [gx, gy], 1e-1, k_size)
			x_i =  coarse_image_est(y_i, k_est, 2e-3)
			xb = shock(x_i,5,1)
		k_max = np.max(np.ravel(k_est));
		k_est[ k_est < 0.05*k_max]  = 0	
		k_est /= np.sum(k_est)

		k_est = kernel_est_l1(xb, y_i, k_est, k_size, k_size, mu = 1e-2)
		k_est = np.clip(k_est,0,np.inf)
		k_est /= np.sum(k_est)

	
	return x_i, k_est

def multiscale_kernel1(y, M, p4ip, denoiser, k_sizes = [7, 13, 25]):
	H, W = np.shape(y)
	
	levels = len(k_sizes)
		
	yn = p4ip_denoiser(y, M, denoiser)
	k_est = np.zeros([7,7], dtype=np.float32); k_est[3,3] = 1.0
	xi = p4ip_wrapper(y, k_est, M, p4ip)
	for iters in range(levels):
		factor = 2**(len(k_sizes)-1-iters)
		H1, W1 = np.int32(H/factor), np.int32(W/factor)
		k_size = k_sizes[iters] 
		k_est = imresize(k_est, [k_size, k_size])
		xi = imresize(xi, [H1, W1])
		yi = imresize(y, [H1, W1])
		yn_i = imresize(yn, [H1, W1])
		# Run an iterative scheme which estimates the kernel and image alternatively
		# Kernel is estimated using TV-L1-prior and image using P4IP solver
		for cc in range(5):
			xb = shock(xi,5,0.1)
			k_est = kernel_est_l1(xb, yn_i, k_est, k_size, k_size, mu = 1e-2)
			k_est= np.clip(k_est,0,np.inf)
			k_est /= np.sum(k_est)
			xi =  p4ip_wrapper(yi, k_est, M, p4ip)
			
	return xi, k_est


def kernel_est_l1(f, g, k0, colh, rowh, mu = 1e-2, rho=20):
	# Solving for ||f*h - g||^2 + ||Dh||_1
	colf, rowf = np.shape(f)
	
	fb = cv.GaussianBlur(f,ksize=(9,9),sigmaX=1,sigmaY=1)
	eye =  np.zeros([3,3], dtype=np.float32); eye[1,1] = 1.0
	dx, dy = D(eye)
	
	fb = cv.GaussianBlur(f,ksize=(9,9),sigmaX=1,sigmaY=1)
	fs = shock(fb, 5, 1)
	Dxfs, Dyfs = D(fs)
	gx, gy = D(g)
	
	_, gxF = psf2otf(gx, [rowf,colf])
	_, gyF = psf2otf(gy, [rowf,colf])
	_, dx_F = psf2otf(dx, [rowf,colf])
	_, dy_F = psf2otf(dy, [rowf,colf])

	_, Dxfs_F = psf2otf(Dxfs, [rowf, colf])
	_, Dyfs_F = psf2otf(Dyfs, [rowf, colf])


	
	num1 = np.conj(Dxfs_F)*gxF
	num2 = np.conj(Dyfs_F)*gyF

	den1 = np.abs(Dxfs_F)**2
	den2 = np.abs(Dyfs_F)**2

	h = k0
	z = k0
	u = np.zeros([rowh, colh])
	
	for ii in range(0, 10):
		# h update
		hhat = z - u 
		_, num3 = psf2otf(hhat, [rowf, colf])
		H = ( (num1 + num2) + rho*num3) / ( den1 + den2 + rho)
		h = np.real(otf2psf(H, [rowh, colh]))
		
		# z update
		zhat  = h + u
		z = np.maximum(np.abs(zhat) - mu/rho, 0) * np.sign(zhat)
		
		u = u + h - z

	h[h < 0] = 0
	h[h < np.max(h) * 0.05] = 0
	h = h / np.sum(h)

	return h

def coarse_image_est(y, k, gamma):
	eye =  np.zeros([3,3], dtype=np.float32); eye[1,1] = 1.0
	dx, dy = D(eye)
	_ , H = psf2otf(k, np.shape(y))
	_, Dx_F = psf2otf(dx, np.shape(y))
	_, Dy_F = psf2otf(dy, np.shape(y))
	num = np.conj(H)*fft2(y)
	den1 = np.abs(H)**2
	den2 = np.abs(Dx_F)**2 + np.abs(Dy_F)**2
	den = den1+gamma*den2
	x_rec = np.real(ifft2(num/den))

	return x_rec

def coarse_image_est_spatial(B, k, Is_list, gamma):
	Ix_s, Iy_s = Is_list[0], Is_list[1]
	eye =  np.zeros([1,1], dtype=np.float32); eye[0,0] = 1.0
	dx, dy = D(eye)

	_ , H = psf2otf(k, np.shape(B))
	_, Dx_F = psf2otf(dx, np.shape(B))
	_, Dy_F = psf2otf(dy, np.shape(B))

	_, Ix_F = psf2otf(Ix_s, np.shape(B))
	_, Iy_F = psf2otf(Iy_s, np.shape(B))

	num1 = np.conj(H)*fft2(B)
	num2 = np.conj(Dx_F)*Ix_F + np.conj(Dy_F)*Iy_F
	den1 = np.abs(H)**2
	den2 = np.abs(Dx_F)**2 + np.abs(Dy_F)**2
	
	num = num1+gamma*num2
	den = den1+gamma*den2
	x_rec = np.real(ifft2(num/den))

	return x_rec

def k_l2(Dx_list, Dy_list, l2_weight, k_size):
	# Solves for k: ||k*fx - gx||^2 + ||k*fy - gy||^2 + l2*||k||^2
	fx, fy = Dx_list[0], Dx_list[1] 
	gx, gy = Dy_list[0], Dy_list[1] 
	k_est_fft =  np.conj(fft2(fx))*fft2(gx) + np.conj(fft2(fy))*fft2(gy) 
	k_est_fft /= (np.abs(fft2(fx)**2) + np.abs(fft2(fy)**2) + l2_weight)
	k_est = otf2psf(k_est_fft, [k_size, k_size])
	
	return k_est

def shrinkage_torch(x, rho):
	return F.relu(x-rho) - F.relu(-x-rho)

class Normalize_Kernel(nn.Module):
	def __init__(self):
		super(Normalize_Kernel, self).__init__()
		self.relu = nn.ReLU()
	def forward(self, k):
		k = self.relu(k) 
		k_sum = torch.sum(k)
		k = k/k_sum
		return k

def show_images(im_list):
	N = len(im_list)

	# Find factors of N
	factors = []
	for i in range(N):
		if N % (i+1) == 0:
			factors.append(i+1)
	N_factors = len(factors)

	A = factors[int(np.floor(N_factors/2))]
	B = int(N/A)
	for idx in range(N):
		plt.subplot(B, A, idx+1)
		plt.imshow(im_list[idx], cmap='gray'); plt.axis('off')
	plt.show()


def coarse_kernel_estimation(y, k_size = 25):
	num_levels = 1
	H, W = np.shape(y)
	k = np.zeros([k_size, k_size], dtype=np.float32)

	# Get a confidence map (r) from blurred image (y)
	Dx, Dy = D(y)
	Dxy = (Dx**2 + Dy**2)**0.5
	fx = cv.blur(Dx,ksize=(k_size,k_size))
	fy = cv.blur(Dy,ksize=(k_size,k_size))
	fxy = cv.blur(Dxy,ksize=(k_size,k_size))
	r = (fx**2 + fy**2)**0.5/(fxy + 0.5)

	# Main loop using multi-resolution scheme starts here
	for idx in reversed(range(num_levels)):
		factor = 2**idx
		H1, W1 = int(H/factor), int(W/factor)
		
		y1 = imresize(y, [H1,W1])
		r1 = imresize(r, [H1,W1])
		if idx == num_levels-1:
			x_est = y1
		else:
			x_est = imresize(x_est, [H1,W1])

		tau_r = 0.01
		tau_s = 0.01
		
		Dx, Dy = D(y1)
		
		M = np.heaviside(r1-tau_r,1.0)
		for _ in range(5):
			# Computing mask M from confidence map
			xb = shock(x_est, 10, 0.1)
			gx, gy = D(xb)
			gxy = (gx**2 + gy**2)**0.5
			M2 = np.heaviside(M*gxy-tau_s,1.0)
			
			# show_images([gx*M2, gy*M2,Dx*M2, Dy*M2])
			k_est = k_l2([gx*M2, gy*M2], [Dx*M2, Dy*M2], 10, k_size)
			k_est = np.clip(k_est,0,np.inf)
			k_est = k_est/np.sum(np.ravel(k_est))
			x_est = coarse_image_est_spatial(y1, k_est, [gx*M2, gy*M2], 2e-3) 
			x_est = np.clip(x_est,0,1)
			# tau_r /= 1.1; tau_s /= 1.1
			# show_images([x_est, k_est])

	return k_est
	
def set_tau_r(r, k_size):
	H, W = np.shape(r)
	N = H*W
	thresh = 0.5*k_size*(N**0.5)
	total_pixels = -np.inf
	tau_r = np.max(r)
	while not total_pixels > thresh:
		tau_r /= 2
		M = np.heaviside(r-tau_r,1.0)
		total_pixels = np.sum(M)
	return tau_r
	
