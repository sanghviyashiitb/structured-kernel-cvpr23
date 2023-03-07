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

	def forward(self, x1, x2, AtA_fft, rho1, rho2):
		lhs = rho1*AtA_fft + rho2 
		rhs = torch.fft.fftn( rho1*x1 + rho2*x2, dim=[2,3] )
		x = torch.fft.ifftn(rhs/lhs, dim=[2,3])
		return x.real

class U_Update(nn.Module):
	def __init__(self):
		super(U_Update, self).__init__()

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


class Z_Update_MIMOUNet(nn.Module):
	def __init__(self):
		super(Z_Update_MIMOUNet, self).__init__()		
		self.net = MIMOUNet() 
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
		outputs = self.net(x.float())
		x_out = outputs[-1]

		return torch.unsqueeze(x_out[:,0,:,:],1)
	
class InitNet(nn.Module):
	def __init__(self, n):
		super(InitNet,self).__init__()
		self.n = n

		self.conv_layers = nn.Sequential(
			Down(1,4),
			Down(4,8),
			Down(8,16),
			Down(16,16))
		
		self.mlp = nn.Sequential(
			nn.Linear(16*8*8+1, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 2*self.n),
			nn.Softplus())
		self.resize = nn.Upsample(size=[256,256], mode='bilinear', align_corners=True)
		
	def forward(self, kernel, M):
		N, C, H, W  = kernel.size()
		h1, h2 = int(np.floor(0.5*(128-H))), int(np.ceil(0.5*(128-H)))
		w1, w2 = int(np.floor(0.5*(128-W))), int(np.ceil(0.5*(128-W)))
		k_pad = F.pad(kernel, (w1,w2,h1,h2), "constant", 0)
		A = torch.fft.fftn(k_pad,dim=[2,3])
		AtA_fft = torch.abs(A)**2
		x = self.conv_layers(AtA_fft.float())
		x = torch.cat((x.view(N,1,16*8*8),  M.float().view(N,1,1)), axis=2).float()
		h = self.mlp(x)+1e-6

		rho1_iters = h[:,:,0:self.n].view(N, 1, 1, self.n)
		rho2_iters = h[:,:,self.n:2*self.n].view(N, 1, 1, self.n)
		return rho1_iters, rho2_iters
		
		
		
class P4IP_Net(nn.Module):
	def __init__(self, n_iters=8, denoiser = 'ResUNet'):
		super(P4IP_Net, self).__init__()
		self.n =  n_iters
		self.init = InitNet(self.n)
		self.X = X_Update() # FFT based quadratic solution
		self.U = U_Update()	# Poisson MLE
		if denoiser == 'ResUNet':
			self.Z = Z_Update_ResUNet() # BW Denoiser
		if denoiser == 'NAFNet':
			self.Z = Baseline_NAFNet() 	# NAFNet baseline architecture
		if denoiser == 'MIMOUNet':
			self.Z = Z_Update_MIMOUNet() 	# NAFNet baseline architecture
	def init_l2(self, y, A, M):
		N, C, H, W = y.size()
		At, AtA_fft = torch.conj(A), torch.abs(A)**2
		rhs = torch.fft.fftn( conv_fft_batch(At, y/M), dim=[2,3] )
		lhs = AtA_fft + (1/M)
		x0 = torch.real(torch.fft.ifftn(rhs/lhs, dim=[2,3]))
		x0 = torch.clamp(x0,0,1)
		return x0

	def forward(self, y, kernel, M):
		z_list = []
		N, C, H, W = y.size()
		# Generate auxiliary variables for convolution
		k_pad, A = psf_to_otf(kernel, y.size())		
		A =  A.to('cuda')
		At, AtA_fft = torch.conj(A), torch.abs(A)**2
		rho1_iters, rho2_iters = self.init(kernel, M) 	# Hyperparameters
		x = self.init_l2(y, A, M)	# Initialization using Weiner Deconvolution
		# Other ADMM variables
		z = Variable(x.data.clone()).to('cuda')
		u = Variable(y.data.clone()).to('cuda')
		v1 = torch.zeros(y.size()).to('cuda')
		v2 = torch.zeros(y.size()).to('cuda')
			
		for n in range(self.n):
			rho1 = rho1_iters[:,:,:,n].view(N,1,1,1)
			rho2 = rho2_iters[:,:,:,n].view(N,1,1,1)
			# U, Z and X updates
			u = self.U(conv_fft_batch(A,x) + v1, y, rho1, M)
			z = self.Z(x.float() + v2.float())
			x = self.X( conv_fft_batch(At,u - v1), z - v2, AtA_fft, rho1, rho2)
			# Lagrangian updates			
			v1 = v1 + conv_fft_batch(A,x) - u
			v2 = v2 + x - z
			z_list.append(z)

		return z_list

