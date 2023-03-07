import torch
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift

from models.ResUNet import ResUNet
from utils.utils_deblur import pad
from utils.utils_torch import conv_fft, conv_fft_batch, psf_to_otf



def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
	elif classname.find('Linear') != -1:
		nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
	elif classname.find('BatchNorm') != -1:
		# nn.init.uniform(m.weight.data, 1.0, 0.02)
		m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
		nn.init.constant(m.bias.data, 0.0)



class DoubleConv(nn.Module):
	"""(convolution => [BN] => ReLU) * 2"""

	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.double_conv(x)

class Down(nn.Module):
	"""Downscaling with maxpool then double conv"""

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)

	def forward(self, x):
		return self.maxpool_conv(x)


class X_Update(nn.Module):
	def __init__(self):
		super(X_Update, self).__init__()

	def forward(self, x, y, rho1, M):
		t1 = rho1*x - M 
		return 0.5*(1/rho1)*( t1 + torch.sqrt( (t1**2)+4*y*rho1)  )

class Z_Update_ResUNet(nn.Module):
	def __init__(self):
		super(Z_Update_ResUNet, self).__init__()		
		self.net = ResUNet() 
		# Load saved weights
		if torch.cuda.is_available():
			self.device = torch.device("cuda:0")
			device_ids = [0]
			model = nn.DataParallel(self.net, device_ids=device_ids).cuda()
		else:
			self.device = torch.device("cpu")
			state_dict = torch.load(model_fn, map_location='cpu')
			# CPU mode: remove the DataParallel wrapper
			state_dict = remove_dataparallel_wrapper(state_dict)
			model = self.net
		
	def forward(self, x):
		x_out = self.net(x.float())
		return x_out
	
class InitNet(nn.Module):
	def __init__(self, n):
		super(InitNet,self).__init__()
		self.n = n
		
		self.mlp = nn.Sequential(
			nn.Linear(1, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, self.n),
			nn.Softplus())
		
	def forward(self, M):
		N = M.size(0)
		h = self.mlp(M.float().view(N,1,1))+1e-6

		rho1_iters = h[:,:,0:self.n].view(N, 1, 1, self.n)
		return rho1_iters
		
		
		
class P4IP_Denoiser(nn.Module):
	def __init__(self, n_iters=8):
		super(P4IP_Denoiser, self).__init__()
		self.n =  n_iters
		self.init = InitNet(self.n)
		self.X = X_Update() 		# Poisson-MLE
		self.Z = Z_Update_ResUNet() # BW Denoiser
	

	def forward(self, y, M):
		x_list = []
		N = y.size(0)
		rho1_iters = self.init(M) 	# Hyperparameters for iteration
		# Initialization and other ADMM variables
		x = y/M	 
		z = Variable(x.data.clone()).to('cuda')
		v1 = torch.zeros(y.size()).to('cuda')
		x_list.append(x)
		for n in range(self.n):
			rho1 = rho1_iters[:,:,:,n].view(N,1,1,1)
			# U, Z and X updates
			x = self.X(z - v1, y, rho1, M)
			z = self.Z(x + v1)
			# Lagrangian updates			
			v1 = v1 + x - z
			x_list.append(x)

		return x_list[-1]
