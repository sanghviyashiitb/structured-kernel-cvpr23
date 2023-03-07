import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2 as cv2
from scipy.interpolate import BSpline, splrep
from torch.utils.data import Dataset, DataLoader

np.random.seed(4)
torch.manual_seed(4)

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


def find_nearest(array, value):
	idx = (np.abs(array - value)).argmin()	
	return array[idx], idx

def get_grid_image(IM_SIZE, SPACE, DOT_SIZE):
	# Make an image consisting of grid points
	grid = np.zeros([IM_SIZE, IM_SIZE], dtype=np.float32)
	for x in np.arange(SPACE, IM_SIZE-1, SPACE):
		for y in np.arange(SPACE, IM_SIZE-1, SPACE):
			grid[x:x+DOT_SIZE,y:y+DOT_SIZE] = 1.0
	return grid

def get_weight_function(trajectory, Tx_arr, Ty_arr, theta_arr):
	N, T = np.shape(trajectory)
		
	TWO_DIMENTIONAL = True if T == 2 else False
	if TWO_DIMENTIONAL:
		trajectory = np.concatenate([trajectory,np.zeros([N,1])],axis=1)
	T = np.shape(trajectory)[0]
	Nx, Ny, Nz = len(Tx_arr), len(Ty_arr), len(theta_arr)
	trajectory_weights = np.zeros([Nx, Ny, Nz])
	for t in range(T):
		x, y, theta = trajectory[t,0], trajectory[t,1], trajectory[t,2]
		_, x_idx = find_nearest(Tx_arr, x)
		_, y_idx = find_nearest(Ty_arr, y)
		_, theta_idx = find_nearest(theta_arr, theta)
		trajectory_weights[x_idx, y_idx, theta_idx ] +=1

	return trajectory_weights

def convert_trajectory_to_kernels(trajectory_3d, grid_in, K_list):
	# IM_SIZE, SPACE, DOT_SIZE = 64, 32, 1
	# F_pixels = 1
	# grid = get_grid_image(IM_SIZE, SPACE, DOT_SIZE)
	# K, K1 = get_camera_calibration(F_pixels, F_pixels, IM_SIZE//2,IM_SIZE//2)
	K, K1 = K_list[0], K_list[1]
	IM_SIZE, _ = np.shape(grid_in)


	Tx_arr = np.linspace(-40,40,100)
	Ty_arr = np.linspace(-40,40,100)

	theta_arr = np.linspace(-10,10,100)

	# Quantize the 3D trajectory to get a weight function
	trajectory_weights = get_weight_function(trajectory_3d, Tx_arr, Ty_arr, theta_arr)
	total_sum = 0 
	grid_out = np.zeros([IM_SIZE, IM_SIZE], dtype=np.float32)
	for i1 in range(len(Tx_arr)):
		for j1 in range(len(Tx_arr)):
			for k1 in range(len(theta_arr)):
				weight = trajectory_weights[i1,j1,k1]
				if weight > 0:
					M = get_transformation_matrix(Tx_arr[i1],Ty_arr[j1],theta_arr[k1],K,K1)
					grid_temp = cv2.warpPerspective(grid_in, M, (IM_SIZE,IM_SIZE))
					grid_out += grid_temp*weight
					total_sum += weight
	
	grid_out /= np.sum(trajectory_weights)
	grid_out = grid_out/np.max(grid_out)
	return grid_out

def get_camera_calibration(fx, fy, x0, y0):
	K = np.zeros([3,3], dtype=np.float32)
	K[0,0], K[1,1], K[2,2] = fx, fy, 1.0
	K[0,2], K[1,2] = x0, y0
	return K, np.linalg.inv(K) 

def get_transformation_matrix(Tx, Ty, theta_z, K, K1):
	# Tx, Ty but theta_z in degrees
	R = np.zeros([3,3], dtype=np.float32)
	cos_theta, sin_theta = np.cos(theta_z*np.pi/180), np.sin(theta_z*np.pi/180)
	R[0,0] = cos_theta; R[1,1] = cos_theta
	R[0,1] = -sin_theta; R[1,0] = sin_theta
	R[0,2] = Tx; R[1,2] = Ty
	R[2,2] =  1.0

	M = np.matmul(np.matmul(K, R), K1)
	return M

def center_trajectory(x):
	if np.ndim(x) == 1:
		x_mean = np.mean(x)
	else:
		x_mean = np.mean(x,axis=0,keepdims=True)
	return x-x_mean


def get_control_points(max_control_points = 5, max_speed = 32):
	N_c = np.random.randint(2,max_control_points+1)
	control_points = np.zeros([N_c, 2], dtype=np.float32)
	for idx in range(1, N_c):
		curr_pos = control_points[idx-1,:]
		theta0, rho = np.random.uniform()*180, max_speed*np.random.uniform()  
		theta = np.pi*theta0/180;
		v_x, v_y = rho*np.cos(theta), rho*np.sin(theta)
		control_points[idx,:] = curr_pos + np.reshape([v_x, v_y], [2])

	return control_points

def create_trajectory(control_points, N):
	x = control_points[:,0]
	y = control_points[:,1]	
	k_hat = min(2, len(x)-1)
	t_hat = np.linspace(1, len(x), len(x) )
	
	t, c, k = splrep(t_hat, x, s=0, k=k_hat)
	spl = BSpline(t, c, k, 2)	
	xx = np.linspace(1,len(x),N)
	x_interp = spl(xx)

	t, c, k = splrep(t_hat, y, s=0, k=k_hat)
	spl = BSpline(t, c, k, 2)	
	yy = np.linspace(1,len(y),N)
	y_interp = spl(yy)

	trajectory = np.zeros([N, 2])
	trajectory[:,0] = np.reshape(x_interp, [N])
	trajectory[:,1] = np.reshape(y_interp, [N])
	return center_trajectory(trajectory)


class Kernels_As_ControlPoints(Dataset):
	def __init__(self, train = True, k_size = 64, n_control_points = 5):
		F_pixels = 1
		self.k_size = k_size
		self.length = 20000
		self.n_control_points = n_control_points
		# parameters required for converting a trajectory to kernel
		self.grid_in = get_grid_image(k_size, k_size//2, 1)
		K, K1 = get_camera_calibration(1,1,k_size//2,k_size//2)
		self.K_list = [K, K1]
		self.train = train
	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		idx1 = idx % self.length
		if self.train:
			idx1 += self.length
		np.random.seed(idx1)
		control_points = get_control_points(self.n_control_points)
		trajectory = create_trajectory(control_points, 500)
		kernel = center_kernel(convert_trajectory_to_kernels(trajectory, self.grid_in, self.K_list))
		# Convert control_points to a standardised vector format
		# Remove the zeros first
		control_points_vec = np.zeros([2*(self.n_control_points-1)], dtype=np.float32)
		# Convert x,y coordinates to 2 X (self.n_control_points-1) X 1 vector
		for idx in range(1,np.shape(control_points)[0]):
			control_points_vec[idx*2-2] = control_points[idx,0]
			control_points_vec[idx*2-1] = control_points[idx,1]
		torch_vec = torch.from_numpy(control_points_vec).view(1,2*self.n_control_points-2)
		kernel = torch.from_numpy(kernel).view(1,self.k_size, self.k_size)
		return [control_points_vec, kernel]

