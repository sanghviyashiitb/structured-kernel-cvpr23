import sys
import os
sys.path.insert(0,'.')

import argparse
import numpy as np
from numpy.linalg import norm
from numpy.fft import fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from PIL import Image
from scipy.io import savemat, loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from utils.utils_deblur import gauss_kernel, pad, crop, psf2otf, otf2psf, D
from utils.utils_torch import conv_fft_batch, psf_to_otf 
from utils.utils_torch import img_to_tens, scalar_to_tens

import DMPHN as models

def shift_inv_psnr(x1, x2, search_window=20):
	max_psnr = -np.inf
	best_match = x1
	for i in np.arange(-search_window, search_window+1):
		for j in np.arange(-search_window, search_window+1):
			x0 = np.roll(x1, (i,j), axis=[0,1])
			mse = np.mean((x0-x2)**2)
			psnr =  -10*np.log10(mse)
			if psnr > max_psnr:
				max_psnr = psnr
				best_match = x0
	return max_psnr, best_match

def shift_inv_metrics(x1, x2, search_window=20):
	max_psnr = -np.inf
	max_ssim = -np.inf
	for i in np.arange(-search_window, search_window+1):
		for j in np.arange(-search_window, search_window+1):
			x0 = np.roll(x1, (i,j), axis=[0,1])
			mse = np.mean((x0-x2)**2)
			psnr =  -10*np.log10(mse)
			ssim_val = ssim(x2, x0, data_range=np.max(np.ravel(x0))-np.min(np.ravel(x0)))

			if psnr > max_psnr:
				max_psnr = psnr
			if ssim_val > max_ssim:
				max_ssim = ssim_val

	return max_psnr, max_ssim

def DMPHN_run(models, img, device): 
	H = img.size(2)          
	W = img.size(3)
	encoder_lv1,encoder_lv2, encoder_lv3, encoder_lv4 = models['encoder']
	decoder_lv1,decoder_lv2, decoder_lv3, decoder_lv4 = models['decoder']
	images_lv1 = img
	images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
	images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
	images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
	images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
	images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
	images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]
	images_lv4_1 = images_lv3_1[:,:,0:int(H/4),:]
	images_lv4_2 = images_lv3_1[:,:,int(H/4):int(H/2),:]
	images_lv4_3 = images_lv3_2[:,:,0:int(H/4),:]
	images_lv4_4 = images_lv3_2[:,:,int(H/4):int(H/2),:]
	images_lv4_5 = images_lv3_3[:,:,0:int(H/4),:]
	images_lv4_6 = images_lv3_3[:,:,int(H/4):int(H/2),:]
	images_lv4_7 = images_lv3_4[:,:,0:int(H/4),:]
	images_lv4_8 = images_lv3_4[:,:,int(H/4):int(H/2),:]

	feature_lv4_1 = encoder_lv4(images_lv4_1)
	feature_lv4_2 = encoder_lv4(images_lv4_2)
	feature_lv4_3 = encoder_lv4(images_lv4_3)
	feature_lv4_4 = encoder_lv4(images_lv4_4)
	feature_lv4_5 = encoder_lv4(images_lv4_5)
	feature_lv4_6 = encoder_lv4(images_lv4_6)
	feature_lv4_7 = encoder_lv4(images_lv4_7)
	feature_lv4_8 = encoder_lv4(images_lv4_8)
	feature_lv4_top_left = torch.cat((feature_lv4_1, feature_lv4_2), 2)
	feature_lv4_top_right = torch.cat((feature_lv4_3, feature_lv4_4), 2)
	feature_lv4_bot_left = torch.cat((feature_lv4_5, feature_lv4_6), 2)
	feature_lv4_bot_right = torch.cat((feature_lv4_7, feature_lv4_8), 2)
	feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
	feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
	feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)
	residual_lv4_top_left = decoder_lv4(feature_lv4_top_left)
	residual_lv4_top_right = decoder_lv4(feature_lv4_top_right)
	residual_lv4_bot_left = decoder_lv4(feature_lv4_bot_left)
	residual_lv4_bot_right = decoder_lv4(feature_lv4_bot_right)

	feature_lv3_1 = encoder_lv3(images_lv3_1 + residual_lv4_top_left)
	feature_lv3_2 = encoder_lv3(images_lv3_2 + residual_lv4_top_right)
	feature_lv3_3 = encoder_lv3(images_lv3_3 + residual_lv4_bot_left)
	feature_lv3_4 = encoder_lv3(images_lv3_4 + residual_lv4_bot_right)
	feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3) + feature_lv4_top
	feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3) + feature_lv4_bot
	feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
	residual_lv3_top = decoder_lv3(feature_lv3_top)
	residual_lv3_bot = decoder_lv3(feature_lv3_bot)

	feature_lv2_1 = encoder_lv2(images_lv2_1 + residual_lv3_top)
	feature_lv2_2 = encoder_lv2(images_lv2_2 + residual_lv3_bot)
	feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
	residual_lv2 = decoder_lv2(feature_lv2)

	feature_lv1 = encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
	out = decoder_lv1(feature_lv1)
	
	return out

def DMPHN_wrapper(y, DMPHN, mode='circular'):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if mode == 'circular':
		yt = img_to_tens(y).to(device)
		with torch.no_grad():
			x_rec = DMPHN_run(DMPHN, yt, device)
		x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)
		x_out = x_rec[0,0,:,:]
	if mode == 'symmetric':
		H, W = np.shape(y); H1, W1 = H//4, W//4
		y_pad = np.pad(y, ((H1,H1), (W1,W1)), mode='symmetric')
		yt = img_to_tens(y).to(device)
		with torch.no_grad():
			x_rec = DMPHN_run(srn, yt, device)
		x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)
		x_out = x_rec[0,0,H1:H1+H,W1:W1+W]
	return x_out

def load_DMPHN(ckpt_path):
	
	encoder_lv1 = models.Encoder()
	encoder_lv2 = models.Encoder()    
	encoder_lv3 = models.Encoder()
	encoder_lv4 = models.Encoder()

	decoder_lv1 = models.Decoder()
	decoder_lv2 = models.Decoder()    
	decoder_lv3 = models.Decoder()
	decoder_lv4 = models.Decoder()

	encoder_lv1.load_state_dict(torch.load(ckpt_path + '/epoch%d/encoder_lv1.pkl'%ckpt_epoch))
	encoder_lv2.load_state_dict(torch.load(ckpt_path + '/epoch%d/encoder_lv2.pkl'%ckpt_epoch))
	encoder_lv3.load_state_dict(torch.load(ckpt_path + '/epoch%d/encoder_lv3.pkl'%ckpt_epoch))
	encoder_lv4.load_state_dict(torch.load(ckpt_path + '/epoch%d/encoder_lv4.pkl'%ckpt_epoch))
	
	decoder_lv1.load_state_dict(torch.load(ckpt_path + '/epoch%d/decoder_lv1.pkl'%ckpt_epoch))
	decoder_lv2.load_state_dict(torch.load(ckpt_path + '/epoch%d/decoder_lv2.pkl'%ckpt_epoch))
	decoder_lv3.load_state_dict(torch.load(ckpt_path + '/epoch%d/decoder_lv3.pkl'%ckpt_epoch))
	decoder_lv4.load_state_dict(torch.load(ckpt_path + '/epoch%d/decoder_lv4.pkl'%ckpt_epoch))
	
	encoder_lv1 = encoder_lv1.to(device).eval()  
	encoder_lv2 = encoder_lv2.to(device).eval()  
	encoder_lv3 = encoder_lv3.to(device).eval()  
	encoder_lv4 = encoder_lv4.to(device).eval()  

	decoder_lv1 = decoder_lv1.to(device).eval()  
	decoder_lv2 = decoder_lv2.to(device).eval()   
	decoder_lv3 = decoder_lv3.to(device).eval()  
	decoder_lv4 = decoder_lv4.to(device).eval()

	DMPHN = {'encoder':(encoder_lv1,encoder_lv2,encoder_lv3,encoder_lv4),'decoder':(decoder_lv1,decoder_lv2,decoder_lv3,decoder_lv4)}
	return DMPHN


np.random.seed(20)
ALPHA = 40
LEVIN_DIR = '../levin/groundtruth/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



psnr_arr = np.zeros([2,4,8], dtype=np.float32)
ssim_arr = np.zeros([2,4,8], dtype=np.float32)
for IMG_IDX in range(1,5):
	for K_IDX in range(1,9): 
		x = np.asarray(Image.open(LEVIN_DIR+'im'+str(IMG_IDX)+'.png'), dtype=np.float32)
		x = x/255.0
		kernel = np.asarray(Image.open(LEVIN_DIR+'kernel'+str(K_IDX)+'.png'), dtype=np.float32)
		kernel = kernel/np.sum(kernel.ravel())		
		yn = np.asarray(Image.open(LEVIN_DIR+'im'+str(IMG_IDX)+'_kernel'+str(K_IDX)+'_img.png'), dtype=np.float32)
		yn = yn/255.0


		if x.ndim > 2:	x = np.mean(x,axis=2)
		if yn.ndim > 2:	yn = np.mean(yn, axis=2)
		# Noisy Image here
		y = np.random.poisson(np.maximum(ALPHA*yn,0)).astype(np.float32)		
		y1 = np.pad(y, ((0,1),(0,1)), mode='symmetric')

		start = time.time()
		
		x_mpr = net_wrapper1(y1/ALPHA, DMPHN)
		
		x_mpr = x_mpr[0:255,0:255]
		stop = time.time()

		# # Calculating shift invariant psnrs
		psnr_mpr, ssim_mpr = shift_inv_metrics(x_mpr, x)

		# # Record PSNRs in an array and save as .npy
		psnr_arr[0,IMG_IDX-1, K_IDX-1] = psnr_mpr
		ssim_arr[0,IMG_IDX-1, K_IDX-1] = ssim_mpr


avg = psnr_arr.mean(axis=(1,2))
avg1 = ssim_arr.mean(axis=(1,2))

print('Photon Level: ', ALPHA)
print('PSNR')
print('DMPHN')
print('%0.2f '%(avg[0]))
print('SSIM')
print('DMPHN')
print('%0.3f '%(avg1[0]))