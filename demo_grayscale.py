import numpy as np
import matplotlib.pyplot as plt
import torch
import argparse
from PIL import Image

from models.phdnet.network_p4ip import P4IP_Net
from models.phdnet.network_p4ip_denoiser import P4IP_Denoiser
from models.ktn.kernel_mlp import Kernel_MLP

from utils.utils_test import add_inset
from utils.utils_deblur import pad
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



if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='deblur arguments')
	parser.add_argument('--ALPHA', type=float, default=20, help='Photon shot noise level')
	args = parser.parse_args()
	ALPHA = args.ALPHA

	np.random.seed(20)
	networks = load_networks()

	x = np.asarray(Image.open('images/levin_clean.png'), dtype = np.float32)/255.0
	yn = np.asarray(Image.open('images/levin_blur.png'), dtype = np.float32)/255.0
	k_true = np.asarray(Image.open('images/levin_kernel.png'), dtype=np.float32)/255.0
	
	y = np.random.poisson(np.maximum(ALPHA*yn,0)).astype(np.float32) # Add synthetic Poisson-shot noise

	opts = {"VERBOSE": True, 'RHO': 1e-4}
	x_list, k_list = iterative_scheme(y, ALPHA, networks, opts)

	k_est = k_list[-1]
	x_est = x_list[-1]

	plt.figure(figsize=(12,6))
	plt.subplot(1,3,1); plt.imshow(y/ALPHA, cmap='gray'); 	
	plt.axis('off'); plt.title('Blurred-Noisy Image')

	plt.subplot(1,3,2); plt.imshow(add_inset(x_est, k_est), cmap='gray'); 	
	plt.axis('off'); plt.title('Restored Image \n (Estimated Kernel in Inset)')
	
	plt.subplot(1,3,3); plt.imshow(add_inset(x, pad(k_true, [64,64])), cmap='gray');
	plt.axis('off'); plt.title('Clean Image \n(True Kernel in Inset)')
	plt.savefig('results/demo_grayscale_output.png', bbox_inches='tight')
	plt.show()