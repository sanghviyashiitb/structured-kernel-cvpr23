import numpy as np
import matplotlib.pyplot as plt

def gauss_kernel(sigma1, sigma2, theta, size):
	cos_theta, sin_theta = np.cos(theta*np.pi/180), np.sin(theta*np.pi/180) 
	ax = np.linspace(-(size-1)*0.5, size*0.5, size)
	xx, yy = np.meshgrid(ax, ax)

	uu = xx*cos_theta - yy*sin_theta
	vv = xx*sin_theta + yy*cos_theta
	kernel = np.exp( -(uu**2)/(2*(sigma1**2)) - (vv**2)/(2*(sigma2**2)) )
	kernel = kernel/np.sum(kernel.ravel())
	return kernel

def pad(h, shape_x):
	shape_h = np.shape(h)
	offset = 1
	hpad = np.zeros(shape_x, dtype=np.float32)
	i1, j1 = np.int32((shape_x[0] - shape_h[0])/2)+offset, np.int32((shape_x[1] - shape_h[1])/2)+offset 
	i2, j2 = i1 + shape_h[0], j1 + shape_h[1]
	hpad[i1:i2, j1:j2] = h
	return hpad

def get_maximum_gradients(Dx, Dy):
	theta_range = np.linspace(0, 179, 180)
	max_gradients = [] 
	for theta in theta_range:
		cos_theta, sin_theta = np.cos(np.pi*theta/180), np.sin(np.pi*theta/180)
		D_theta = np.max(np.abs(Dx*cos_theta + Dy*sin_theta))
		max_gradients.append(D_theta)       
	return max_gradients

def get_theta(max_gradients):
	theta_max = np.argmin(max_gradients)
	theta_ortho = np.mod(90+theta_max,180) 
	
	f_theta = max_gradients[theta_max]
	f_theta_ortho = max_gradients[theta_ortho]
	return theta_ortho, f_theta, f_theta_ortho

def estimate_gaussian_kernel(im, k_size = 35, C=101.0/255.0, SIGMA_B = 0.764):
	Dx, Dy = np.gradient(im)
	max_gradients = get_maximum_gradients(Dx, Dy)
	theta, f_theta, f_theta_ortho = get_theta(max_gradients)
	sigma_0 = np.sqrt( (C/f_theta)**2 - SIGMA_B**2)
	sigma_1 = np.sqrt( (C/f_theta_ortho)**2 - SIGMA_B**2)
	
	kernel = gauss_kernel(sigma_0, sigma_1, theta, k_size)
	gauss_params = [sigma_0, sigma_1, theta]
	return kernel, gauss_params


# max_gradients = get_maximum_gradients(Dx, Dy)
# kernel_est, sigma_0_est, sigma_1_est, theta_est = estimate_gaussian_kernel(max_gradients)
