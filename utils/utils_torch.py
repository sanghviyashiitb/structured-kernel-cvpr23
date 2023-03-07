import numpy as np
from PIL import Image
from utils.utils_deblur import gauss_kernel, pad, crop
from numpy.fft import fft2
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# functionName implies a torch version of the function
def fftn(x):
	x_fft = torch.fft.fftn(x,dim=[2,3])
	return x_fft

def ifftn(x):
	return torch.fft.ifftn(x,dim=[2,3])

def ifftshift(x):
	# Copied from user vmos1 in a forum - https://github.com/locuslab/pytorch_fft/issues/9
	for dim in range(len(x.size()) - 1, 0, -1):
		x = torch.roll(x, dims=dim, shifts=x.size(dim)//2)
	return x

def conv_fft(H, x):
	if x.ndim > 3: 
		# Batched version of convolution
		Y_fft = fftn(x)*H.repeat([x.size(0),1,1,1])
		y = ifftn(Y_fft)
	if x.ndim == 3:
		# Non-batched version of convolution
		Y_fft = torch.fft.fftn(x, dim=[1,2])*H
		y = torch.fft.ifftn(Y_fft, dim=[1,2])
	return y.real

def conv_fft_batch(H, x, mode='circular'):
	# Batched version of convolution
	if mode == 'circular':
		Y_fft = fftn(x)*H
		y = ifftn(Y_fft)
		return y.real
	else:
		_, _ , h, w = x.size()
		_, _, h1, w1 = H.size()
		h2, w2 = h//4, w//4
		m = nn.ReflectionPad2d( (h2,h2,w2,w2) )	
		x_pad = m(x)
		Y_fft = fftn(x_pad)*H
		y = ifftn(Y_fft)
		return y.real[:,:,h2:h+h2,w2:w+w2]



def scalar_to_tens(x):
	return torch.Tensor([x]).view(1,1,1,1)

def conv_kernel(k, x, mode='cyclic'):
	_ , h, w = x.size()
	h1, w1 = np.shape(k)
	k = torch.from_numpy(np.expand_dims(k,0))
	k_pad, H = psf_to_otf(k.view(1,1,h1,w1), [1,1,h,w])
	H = H.view(1,h,w)
	Ax = conv_fft(H,x)

	return Ax, k_pad.view(1,h,w)

def conv_kernel_symm(k, x):
	_ , h, w = x.size()
	h1, w1 = np.int32(h/2), np.int32(w/2)
	m = nn.ReflectionPad2d( (h1,h1,w1,w1) )
	x_pad = m(x.view(1,1,h,w)).view(1,h+2*h1, w+2*w1)
	k_pad = torch.from_numpy(np.expand_dims(pad(k,[h+2*h1,w+2*w1]),0))
	H = torch.fft.fftn(k_pad, dim=[1,2])
	Ax_pad = conv_fft(H,x_pad)
	Ax = Ax_pad[:,h1:h+h1,w1:w+w1]
	return Ax, k_pad

def psf_to_otf(ker, size):
    if ker.shape[2] % 2 == 0:
    	ker = F.pad(ker, (0,1,0,1), "constant", 0)
    psf = torch.zeros(size)
    # circularly shift
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf
    # otf = torch.rfft(psf, 3, onesided=False)
    otf = torch.fft.fftn(psf, dim=[2,3])
    return psf, otf

def p4ip_wrapper(y, k, M, p4ip, mode ='circular'):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if mode == 'symmetric':
		H, W = np.shape(y)
		H1, W1 = H//2, W//2
		y = np.pad(y, ((H1,H1),(W1,W1)), mode='symmetric')
	
	Ht = img_to_tens(k).to(device).float()	
	yt = img_to_tens(y).to(device)
	Mt = scalar_to_tens(M).to(device)
	with torch.no_grad():
		x_rec_list = p4ip(yt, Ht, Mt)
	x_rec = x_rec_list[-1]
	x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)	
	x_out = x_rec[0,0,:,:]
	if mode == 'symmetric':
		x_out = x_out[H1:H+H1, W1:W+W1]

	return x_out

	

def unet_wrapper(y, unet):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	yt = img_to_tens(y).to(device)
	with torch.no_grad():
		x_rec = unet(yt)
	x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)
	x_out = x_rec[0,0,:,:]
	return x_out

def net_wrapper(y, srn):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	yt = img_to_tens(y).to(device)
	with torch.no_grad():
		x_rec = srn(yt)
	if isinstance(x_rec, list):
		x_out = x_rec[-1]
	else:
		x_out = x_rec
	x_out = np.clip(x_out.cpu().detach().numpy(),0,1)
	x_out = x_out[0,0,:,:]
	return x_out



def p4ip_denoiser(y, M, denoiser):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	yt = img_to_tens(y).to(device)
	Mt = scalar_to_tens(M).to(device)
	with torch.no_grad():
		x_rec = denoiser(yt, Mt)
	if isinstance(x_rec, list):
		x_rec = x_rec_list[-1]
	x_rec = np.clip(x_rec.cpu().detach().numpy(),0,1)
	x_out = x_rec[0,0,:,:]
	return x_out

def L2_Loss_Gradient(x, y):
	Dx_x, Dy_x = torch.gradient(x, dim=[2,3])
	Dx_y, Dy_y = torch.gradient(y, dim=[2,3])
	L2_LOSS = nn.MSELoss()
	return L2_LOSS(Dx_x, Dx_y) + L2_LOSS(Dy_x, Dy_y)


def sharpness(x):
	Dx, Dy = torch.gradient(x, dim=[2,3])
	return ((Dx.pow(2) + Dy.pow(2)).pow(0.5)).mean()

	return x0

def tens_to_img(xt, size=None):
	if size is None:
		return np.squeeze(np.squeeze(xt.detach().cpu().numpy()))
	else:
		x_np = np.squeeze(np.squeeze(xt.detach().cpu().numpy()))
		return np.reshape(x_np, size)

def img_to_tens(x, size=None):
	xt = torch.from_numpy(np.expand_dims( np.expand_dims(x,0),0))
	if size is None:
		return xt
	else:
		return xt.view(size)

def nan(t):
	return torch.isnan(t).any().item()

class Normalize_Kernel(nn.Module):
	def __init__(self):
		super(Normalize_Kernel, self).__init__()
		self.relu = nn.ReLU()
	def forward(self, k):
		k = self.relu(k) 
		k_sum = torch.sum(k)
		k = k/k_sum
		return k

def shrinkage_torch(x, rho):
	return F.relu(x-rho) - F.relu(-x-rho)


# def Kernel_L1_Loss(k_out, k_target):
# 	# Make sure the k_target is of size k_out
# 	B, _, H, W = k_out.size()
# 	_, _, H1, W1 = k_target.size()
# 	H2, W2 = H//2, W//2
# 	H3, W3 = H1//2, W1//2
# 	if H < H1:
# 		# Cropping out the center of the k_target
# 		k_target = k_target[:,:,H3-H2:H3+H2, W3-W2:W3+W2]
# 		# Making sure the kernel is at the center
		