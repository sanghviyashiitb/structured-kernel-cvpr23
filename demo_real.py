import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from PIL import Image

from models.phdnet.network_p4ip import P4IP_Net
from models.phdnet.network_p4ip_denoiser import P4IP_Denoiser
from models.ktn.kernel_mlp import Kernel_MLP

from utils.utils_test import add_inset, bayer_to_rggb, rggb_to_rgb
from utils.utils_deblur import pad
from utils.utils_torch import p4ip_wrapper
from utils.iterative_scheme import iterative_scheme

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")








def load_networks():
	# Non-Blind Solver, Core-of the iterative scheme, referred to as F(., .) in the paper
	nb_solver = P4IP_Net(); 
	nb_solver.load_state_dict(torch.load('model_zoo/p4ip_latest_resunet.pth'))
	nb_solver.to(device); nb_solver.eval()
	# Blur+Noise -> Blur-only denoiser, referred to as G(.) in the paper
	denoiser = P4IP_Denoiser(); 
	denoiser.load_state_dict(torch.load('model_zoo/denoiser_p4ip_100epoch.pth')); 
	denoiser.to(device); denoiser.eval()
	# Kernel Trajectory Network (KTN), converts latent representation to kernel T(.)
	N_CONTROL = 8
	ktn = Kernel_MLP(N_CONTROL); 
	ktn.load_state_dict(torch.load('model_zoo/kernel_mlp'+str(N_CONTROL)+'_latest.pth'))
	ktn.to(device); ktn.eval()
	networks = {'nb_solver': nb_solver, 'denoiser': denoiser, 'kernel_mlp': ktn}
	return networks



def load_data():
	# IDX = 10
	# DIR_IDX = 5; 
	# IDX_CLEAN = int(IDX/3)

	y = np.load('images/real_noisy_blur.npy')
	k3 = np.clip(np.load('images/real_kernel.npy'),0,np.inf)
	k3 /= np.sum(np.ravel(k3))
	kernel = pad(k3, [64,64])
	y = np.asarray(y, dtype=np.float32)
	
	y_list = bayer_to_rggb(y);
	
	H1, W1, H, W = 400, 300, 256, 256
	y1_list = []
	for y in y_list:
		y1 = np.clip(y[H1:H1+H, W1:W+W1],0,np.inf)
		y1_list.append(y1)
	return y1_list, kernel 
	



if __name__ == "__main__":

	networks = load_networks()
	# Load real-noise data from .npy file in form of RGGB images, and ground-truth kernel
	y1_list, k_true = load_data()
	
	# For real-sensor data, we convert take the average of the RGGB channels as the input
	y_mean = y1_list[0]*0
	for y1 in y1_list:
		y_mean += y1/len(y1_list)

	opts = {'VERBOSE': True, 'USE_GRADIENT_LOSS': True}
	_, k_list = iterative_scheme(y_mean, np.mean(y_mean)/0.33, networks, opts)
	k_est = k_list[-1]

	x_out_list = []
	y_out_list = []
	for y in y1_list:
		H, W = np.shape(y)
		H3, W3 = H//4, W//4
		M = np.mean(y)/0.33
		y_pad = np.pad(y, ((H3,H3),(W3,W3)), mode='reflect')
		x_pad = p4ip_wrapper(y_pad, k_est, M, networks['nb_solver'])
		
		y_out_list.append(y/M)
		x_out_list.append(x_pad[H3:H+H3, W3:W+W3])

	y_rgb = rggb_to_rgb(y_out_list, [1,1,1])
	x_rgb = rggb_to_rgb(x_out_list, [1,1,1])
	
	plt.figure(figsize=(12,6))
	plt.subplot(1,2,1)
	plt.imshow(y_rgb); plt.axis('off')
	plt.title('Noisy-Blur Image')

	plt.subplot(1,2,2)
	plt.imshow(x_rgb); plt.axis('off')
	plt.title('Reconstructed Image')
	plt.show()