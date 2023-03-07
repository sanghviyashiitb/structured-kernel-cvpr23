import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image


def pad(h, shape_x):
	shape_h = np.shape(h)
	offset = 1
	hpad = np.zeros(shape_x, dtype=np.float32)
	i1, j1 = np.int32((shape_x[0] - shape_h[0])/2)+offset, np.int32((shape_x[1] - shape_h[1])/2)+offset 
	i2, j2 = i1 + shape_h[0], j1 + shape_h[1]
	hpad[i1:i2, j1:j2] = h
	return hpad

def shrinkage(z, beta):
	c1, c2 = z -beta,  z + beta
	z_out = np.clip(c1,0,np.inf) + np.clip(z+beta,-np.inf,0)
	return z_out

def crop(h, shape_crop):
	shape_h = np.shape(h)
	i1, j1 = np.int32((shape_h[0] - shape_crop[0])/2), np.int32((shape_h[1] - shape_crop[1])/2) 
	i2, j2 = np.int32((shape_h[0] + shape_crop[0])/2), np.int32((shape_h[1] + shape_crop[1])/2) 
	return h[i1:i2, j1:j2]


def gauss_kernel(size, sigma):
	ax = np.linspace(-(size-1)*0.5, size*0.5, size)
	xx, yy = np.meshgrid(ax, ax)

	kernel = np.exp( -(xx**2 + yy**2)/(2*(sigma**2)) )
	kernel = kernel/np.sum(kernel.ravel())
	return kernel

def disk(size, r):
	ax = np.linspace(-(size-1)*0.5, size*0.5)
	xx, yy = np.meshgrid(ax, ax)

	kernel = np.asarray((xx**2 + yy**2) < r**2, dtype=np.float32)
	kernel = kernel/np.sum(kernel.ravel())
	return kernel

def mask_gradients(D1, D2, tau_s =0.1, tau_r =0.1):
	g = (1/25)*np.ones([5,5], dtype=np.float32)

	Dr = np.sqrt(D1**2 + D2**2)
	a, b = convolve2d(D1,g,mode='same'), convolve2d(D2,g,mode='same')
	c = convolve2d(Dr,g,mode='same')
	R = np.sqrt(a**2 + b**2)/(c +0.5) 
	M = np.max(R-tau_r, 0)
	D11, D21 = D1*np.max(M*Dr-tau_s,0), D2*np.max(M*Dr-tau_s,0)
	return  D11, D21, M

def rgb_to_bayer(x):
	H, W, _ = np.shape(x)
	x_bayer = np.zeros([2*H, 2*W])
	x_r, x_g, x_b =  x[:,:,0], x[:,:,1], x[:,:,2]
	
	x_bayer[0::2,0::2] = x_r
	x_bayer[0::2,1::2] = x_g
	x_bayer[1::2,0::2] = x_g
	x_bayer[1::2,1::2] = x_b

	return x_bayer

def rggb_to_rgb(x_list, switch_rgb):
	H, W = np.shape(x_list[0])
	x_rgb = np.zeros([H, W, 3])
	
	x_rgb[:,:,0] = x_list[0]
	x_rgb[:,:,1] = (x_list[1]+x_list[2])*0.5
	x_rgb[:,:,2] = x_list[3]
	
	if switch_rgb:
		x_rgb = np.flip(x_rgb,2)
	return x_rgb



# def psf2otf(kernel, size):
# 	psf = np.zeros(size,dtype=np.float32)
# 	centre =  np.shape(kernel)[0]//2 + 1
	
# 	psf[:centre, :centre] = kernel[(centre-1):,(centre-1):]
# 	psf[:centre, -(centre-1):] = kernel[(centre-1):, :(centre-1)]
# 	psf[-(centre-1):, :centre] = kernel[:(centre-1), (centre-1):]
# 	psf[-(centre-1):, -(centre-1):] = kernel[:(centre-1),:(centre-1)]
	
# 	otf = fft2(psf, size)
# 	return psf, otf

def psf2otf(psf, outsize):
    psfsize = np.shape(psf)
    psf = np.pad(psf, ( (0,outsize[0]-psfsize[0]), (0,outsize[1]-psfsize[1])),'constant', constant_values=(0,0))

    shiftvalue0 = int(np.floor(psfsize[0]/2))
    shiftvalue1 = int(np.floor(psfsize[1] / 2))

    psf = np.roll(psf,-shiftvalue0,axis=0)
    psf = np.roll(psf, -shiftvalue1, axis=1)

    otf = np.fft.fftn(psf)

    return psf, otf

def otf2psf(otf, size):
	psf = np.real(ifft2(otf))
	psf = np.roll(psf, int(np.floor(size[0]//2)), axis=0)
	psf = np.roll(psf, int(np.floor(size[1]//2)), axis=1)
	psf = psf[0:size[0], 0:size[1]]
	return psf.astype(np.float32)


def otf2psf1(otf, size):
	psf = np.real(ifft2(otf))
	psf = np.roll(psf, int(np.floor(size[0]//2))-1, axis=0)
	psf = np.roll(psf, int(np.floor(size[1]//2))-1, axis=1)
	psf = psf[0:size[0], 0:size[1]]
	return psf.astype(np.float32)


def shock(I0,iter,dt):
	h = 1

	I = I0.copy()

	ss = np.shape(I0)

	for i in range(0,iter):
		Itempmx = I.copy()
		Itempmx[:,1:ss[1]] = Itempmx[:,0:ss[1]-1].copy()
		I_mx = I - Itempmx

		Itemppx = I.copy()
		Itemppx[:,0:ss[1]-1] = Itemppx[:,1:ss[1]].copy()
		I_px = Itemppx - I

		Itempmy = I.copy()
		Itempmy[1:ss[0],:] = Itempmy[0:ss[0] - 1,:].copy()
		I_my = I - Itempmy

		Itemppy = I.copy()
		Itemppy[0:ss[0] - 1,:] = Itemppy[1:ss[0],:].copy()
		I_py = Itemppy - I

		I_x = (I_mx + I_px) / 2
		I_y = (I_my + I_py) / 2

		#minmod operator
		Dx = np.minimum(np.abs(I_mx), np.abs(I_px))
		Dx[I_mx*I_px < 0] = 0

		Dy = np.minimum(np.abs(I_my), np.abs(I_py))
		Dy[I_my * I_py < 0] = 0

		I_xx = Itemppx + Itempmx - 2*I
		I_yy = Itemppy + Itempmy - 2*I

		Itempxy1 = I_x.copy()
		Itempxy1[0: ss[0] - 1,:] = Itempxy1[1:ss[0], :].copy()

		Itempxy2 = I_x.copy()
		Itempxy2[1:ss[0],:] = Itempxy2[0:ss[0] - 1,:].copy()

		I_xy = (Itempxy1 - Itempxy2)/2

		#compute flow
		a_grad_I = np.sqrt(Dx**2 + Dy**2)
		dl = 1e-8
		I_nn = I_xx * (np.abs(I_x)**2) + 2*I_xy * I_x * I_y + I_yy * (np.abs(I_y)**2)
		I_nn = I_nn/((np.abs(I_x)**2) + (np.abs(I_y)**2) + dl)

		I_ee = I_xx * (np.abs(I_y)**2) - 2*I_xy * I_x * I_y + I_yy * (np.abs(I_x)**2)
		I_ee = I_ee / ((np.abs(I_x) ** 2) + (np.abs(I_y) ** 2) + dl)

		a2_grad_I = np.abs(I_x) + np.abs(I_y)

		I_nn[a2_grad_I == 0] = I_xx[a2_grad_I == 0]
		I_ee[a2_grad_I == 0] = I_yy[a2_grad_I == 0]

		I_t = -np.sign(I_nn)*a_grad_I/h

		I = I + dt*I_t

	return I

def D(x):
	H, W = np.shape(x)

	dx1, dx2 = np.diff(x, axis=1), np.reshape( x[:,0]-x[:,-1], [H,1])
	Dx = np.concatenate((dx1,dx2), axis=1)
	dy1, dy2 = np.diff(x, axis=0), np.reshape( x[0,:]-x[-1,:], [1,W])
	Dy = np.concatenate((dy1,dy2), axis=0)

	return Dx, Dy

def Dt(Dx, Dy):
	H, W = np.shape(Dx)
	
	dtx1, dtx2 = np.reshape(Dx[:,-1]-Dx[:,0], [H,1]), -np.diff(Dx, axis=1)
	dty1, dty2 = np.reshape(Dy[-1,:]-Dy[0,:], [1,W]), -np.diff(Dy, axis=0)
	Dt_XY = np.concatenate((dtx1,dtx2), axis=1)
	Dt_XY += np.concatenate((dty1,dty2), axis=0)
	return Dt_XY

def imresize(im, size):
	H, W = size[0], size[1]
	im_PIL = np.array(Image.fromarray(im).resize((W, H), Image.BILINEAR) )
	return im_PIL.astype(np.float32)

def get_first_moments(im):
	rows, cols = np.shape(im)
	seq1 = np.repeat(np.reshape(np.arange(rows), [rows,1]), cols, axis=1)
	seq2 = np.repeat(np.reshape(np.arange(cols), [1,cols]), rows, axis=0)
	mx, my = np.mean(seq1*im)/np.mean(im), np.mean(seq2*im)/np.mean(im)
	return mx, my

def center_kernel(kernel):
	N, _ = np.shape(kernel)
	# Center the image
	mx, my = get_first_moments(kernel)
	shift_x, shift_y =  N//2-np.int32(mx), N//2-np.int32(my)
	kernel = np.roll(kernel, (shift_x, shift_y), axis=[0,1])
	return kernel
	
