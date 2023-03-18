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


class Kernels_As_ControlPoints(Dataset):
	def __init__(self, train = True, k_size = 64, n_control_points = 4):
		self.k_size = k_size
		self.n_control_points = n_control_points
		if train:	
			self.n_files = 10
			self.length_file = 4000
			self.mode = 'train'
		else:
			self.n_files = 5
			self.length_file = 1000
			self.mode = 'val'
			
	def __len__(self):
		return self.length_file*self.n_files
		
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		idx_file = idx // self.length_file; idx_file += 1	
		idx1 = idx % self.length_file; 
		data = np.load('kernel_data_'+self.mode+'_'+str(idx_file)+'_'+str(self.n_control_points)+'.npz')
		vec_list = data['vec_list']; k_list = data['k_list']
		torch_vec = torch.from_numpy(vec_list[idx1,:].astype(np.float32)).view(1,2*(self.n_control_points-1))
		kernel = k_list[idx1,:,:]
		if self.mode == 'val':
			kernel = kernel/np.sum(np.ravel(kernel))
		kernel = torch.from_numpy(kernel.astype(np.float32)).view(1,self.k_size, self.k_size)

		return [torch_vec, kernel]

