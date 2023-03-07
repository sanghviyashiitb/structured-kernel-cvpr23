import sys
sys.path.insert(0,'.')

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2
from PIL.Image import fromarray, open
from os import listdir
from os.path import isfile, join
import torch
from torchvision.transforms import Compose, RandomCrop, Normalize, Grayscale, ToTensor
from torch.utils.data import Dataset, DataLoader
from utils.utils_deblur import gauss_kernel, pad, crop
from utils.utils_torch import conv_kernel, conv_kernel_symm
from utils.motion_blur import Kernel
from torchvision.transforms import Compose, Resize, RandomResizedCrop, Grayscale, ToTensor, RandomVerticalFlip
from scipy.io import loadmat
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
	

class Flickr2K(Dataset):
	def __init__(self, train, transform_img, transform_blur, get_kernel = True):
		self.shuffle  = True
		self.train = train
		if self.train:
			self.root_dirs = ['../P4IP/Python/data/Flickr2K/train']
		else:
			self.root_dirs = ['../P4IP/Python/data/Flickr2K/val']

		self.list_files = []
		for directory in self.root_dirs:
			for f in listdir(directory):
				if isfile(join(directory,f)):
					self.list_files.append(join(directory,f))

		self.transform_img = transform_img
		self.transform_blur = transform_blur
		self.get_kernel = get_kernel
	def __len__(self):
		return len(self.list_files)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		img_name = join(self.list_files[idx])
		img = open(img_name)

		if self.transform_img:
			x = self.transform_img(img)

		if self.transform_blur:
			y, kernel, M = self.transform_blur(x)
			if self.get_kernel:
				sample = [x, y, kernel, M]
			else:
				sample = [x, y, M]
			return sample
		else:
			return x

class Flickr2K_AWGN(Dataset):
	def __init__(self, train, transform_img, noise_range):
		self.shuffle  = True
		self.train = train
		if self.train:
			self.root_dirs = ['../P4IP/Python/data/Flickr2K/train']
		else:
			self.root_dirs = ['../P4IP/Python/data/Flickr2K/val']

		self.list_files = []
		for directory in self.root_dirs:
			for f in listdir(directory):
				if isfile(join(directory,f)):
					self.list_files.append(join(directory,f))

		self.transform_img = transform_img
		self.sigma0 = noise_range[0]
		self.sigma1 = noise_range[1]

	def __len__(self):
		return len(self.list_files)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		img_name = join(self.list_files[idx])
		img = open(img_name)

		if self.transform_img:
			x = self.transform_img(img)

		sigma = np.random.uniform(self.sigma0, self.sigma1)
		y = x + sigma*torch.randn(x.size())
		sample = [x, y, sigma]
		return sample



class PoissBlur_List(object):
	# From the list of given scaling factors, kernels
	# Choose a random scaling size and kernel fft
	# and apply the corresponding transform to image 
	def __init__(self, kernel_list, M_range, N, biased_sampling=True):
		self.M1, self.M2 = M_range[0], M_range[1]
		self.kernel_list = kernel_list
		self.N = N
		self.biased_sampling= biased_sampling
	

	def __call__(self, x):
		if self.biased_sampling:
			M = self.M1 + (self.M2-self.M1)*(np.random.uniform()**1.5)
		else:
			M = np.random.uniform(self.M1, self.M2)
		idx = np.random.choice(np.arange(0,len(self.kernel_list)))
		kernel = self.kernel_list[idx]

		kernel = kernel/np.sum(np.ravel(kernel))
		kernel = center_kernel(kernel)
		kernel_torch = torch.from_numpy(np.expand_dims(kernel,0))
		Ax, k_pad = conv_kernel_symm(kernel, x)
		Ax_min = torch.clamp(Ax,min=1e-6)
		y = torch.poisson(Ax_min*M)
		return y, kernel_torch, M


class PoissBlur_Spatial(object):
	# From the list of given scaling factors, kernels
	# Choose a random scaling size and kernel fft
	# and apply the corresponding transform to image 
	def __init__(self, M_range, N, shuffle_kernel = False):
		self.M1, self.M2 = M_range[0], M_range[1]
		self.kernel_list = Homography_Kernels()
		self.N = N
		self.shuffle_kernel = shuffle_kernel
	def __call__(self, x):
		M = np.random.uniform(self.M1, self.M2)
		
		idx_k = np.random.randint(len(self.kernel_list))
		k1, k2 = self.kernel_list[idx_k]
		# Convolve using k1, but get k2 in return
		k1 = center_kernel(k1)
		k1_torch = torch.from_numpy(np.expand_dims(k1,0))
		k2_torch = torch.from_numpy(np.expand_dims(k2,0))
		Ax, k1_pad = conv_kernel(k1, x)
		Ax_min = torch.clamp(Ax,min=1e-6)
		y = torch.poisson(Ax_min*M)
		
		# Toss a coin to see if you want to shuffle the kernel given
		if self.shuffle_kernel:
			if np.random.uniform() < 0.5:
				k_out = k1_torch
			else:
				k_out = k2_torch
		else:
			k_out = k1_torch	
		return y, k_out, M

class GoPro(Dataset):
	def __init__(self, train = True, transform_noise = None):
		self.shuffle  = True
		self.train = train
		if self.train:
			self.root_dirs = ['../datasets/GOPRO_Large/train/']
		else:
			self.root_dirs = ['../datasets/GOPRO_Large/test/']

		self.list_files = []
		for directory in self.root_dirs:
			for folder in listdir(directory):
				for file in listdir(directory+folder+'/blur'):					
					blur_file = directory+folder+'/blur/'+file
					sharp_file = directory+folder+'/sharp/'+file
					self.list_files.append([sharp_file, blur_file])

		self.transform_noise = transform_noise
		self.ps = 128
	def __len__(self):
		return len(self.list_files)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		
		img_pair = self.list_files[idx]
		img_sharp, img_blur = open(img_pair[0]).resize((512,512)), open(img_pair[1]).resize((512,512))
		
		# Convert to grayscale ->Tensor
		img_sharp = img_sharp.convert("L")
		img_blur = img_blur.convert("L")
		x = torch.from_numpy(np.asarray(img_sharp, dtype=np.float32)/255.0)#.view(1,1,128,128)
		yn = torch.from_numpy(np.asarray(img_blur, dtype=np.float32)/255.0)#.view(1,1,128,128)
		# Choose random crop		
		h, w = np.random.randint(1,self.ps), np.random.randint(1,self.ps)
		x = x[h:h+self.ps,w:w+self.ps].view(1,self.ps,self.ps) 
		yn = yn[h:h+self.ps,w:w+self.ps].view(1,self.ps,self.ps)
		
		if self.transform_noise:
			y, M = self.transform_noise(yn)
			sample = [x, yn, y, M]
			return sample
		else:
			sample = [x, y]
			return sample

class Homography_Kernels(Dataset):
	def __init__(self, train = True, shuffle = False):
		self.shuffle  = shuffle
		self.train = train
		if self.train:
			self.root_dirs = 'data/spatial_varying_kernels/'
		else:
			self.root_dirs = 'data/spatial_varying_kernels/val/'
		

		self.list_files = []
		for idx in range(len(listdir(self.root_dirs))-1):					
			kernel_file = self.root_dirs+'kernel_grid_'+str(idx+1)+'.png'
			traj_file = self.root_dirs+'trajectory_'+str(idx+1)+'.png'
			self.list_files.append([kernel_file, traj_file])
		self.crop_kernel = 64
		
	def __len__(self):
		return len(self.list_files)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		crop = self.crop_kernel//2
		img_pair = self.list_files[idx]
		img_kernel = open(img_pair[0])

		grid = np.asarray(img_kernel, dtype=np.float32)/255.0
		# Choose a 2 X 2 quadrant in the image 
		h1, w1 = np.random.randint(4), np.random.randint(4)
		grid_q = grid[h1*256:h1*256+256,w1*256:w1*256+256]

		# and then two random kernels within the quadrant 
		h2, w2 = np.random.randint(2), np.random.randint(2)
		h3, w3 = np.random.randint(2), np.random.randint(2)
		
		k1 = grid_q[h2*128:(h2+1)*128, w2*128:(w2+1)*128]
		k2 = grid_q[h3*128:(h3+1)*128, w3*128:(w3+1)*128]
		k1 = np.clip(k1, 0, np.inf); k1 = k1/np.sum(k1)
		k2 = np.clip(k2, 0, np.inf); k2 = k2/np.sum(k2)

		k1_crop = k1; k2_crop = k2
		# See if the kernels can be cropped or not
		# for crop in [2,4,8,16,32]:
		# 	if np.sum(k1_crop) < 1 or np.sum(k2_crop) < 1:
		# 		break
		# 	else:
		# 		k1_crop = k1[crop:128-crop,crop:128-crop]
		# 		k2_crop = k2[crop:128-crop,crop:128-crop]

		return k1_crop, k2_crop

class Poiss_List(object):
	# From the list of given scaling factors, kernels
	# Choose a random scaling size and kernel fft
	# and apply the corresponding transform to image 
	def __init__(self, M_range):
		self.M1, self.M2 = M_range[0], M_range[1]
		

	def __call__(self, y):
		M = np.random.uniform(self.M1, self.M2)
		y_noisy = torch.poisson(y*M)
		return y_noisy, M

def get_dataloader(mode = 'synthetic', train = True, get_kernel = False):
	if mode == 'synthetic':
		# Adding Blur Kernels
		kernel_list = []
		struct = loadmat('data/motion_kernels.mat')
		motion_kernels = struct['PSF_list'][0]
		for idx in range(len(motion_kernels)):
			kernel = motion_kernels[idx]
			kernel = np.clip(kernel,0,np.inf)
			kernel = kernel/np.sum(kernel.ravel())
			kernel_list.append(kernel)
		if train == True:
			N_TRAIN = 128
			N_VAL = 256
			transform_img =  Compose([Resize([512,512]),
									RandomCrop([N_TRAIN,N_TRAIN]), 
									Grayscale(), 
									RandomVerticalFlip(), 
									ToTensor()])
			transform_blur = PoissBlur_List(kernel_list, [1,60], N_TRAIN, True)
		else:
			N_VAL = 256
			transform_img =  Compose([Resize([N_VAL,N_VAL]), 
									Grayscale(), 
									ToTensor()])
			transform_blur = PoissBlur_List(kernel_list, [1,60], N_VAL, False)
		# Dataloaders
		data_loader = Flickr2K(train, transform_img, transform_blur, get_kernel)

	if mode == 'real':
		transform_noise = Poiss_List([1,60])
		data_loader = GoPro(train, transform_noise)

	return data_loader


class EnhancementData(Dataset):
	def __init__(self, train = True, transform_noise = None):
		self.train = train
		
		self.directory = 'data/enhancement_data/'
		self.list_files = []
		for file in listdir(self.directory):					
			self.list_files.append(self.directory+file)		
		# Shuffle data files and split into 30:70 val/train split	
		self.list_train = self.list_files
		
	def __len__(self):
		return len(self.list_train)


	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_file = self.list_train[idx]
		struct = loadmat(img_file)
		
		x = torch.unsqueeze(torch.from_numpy(struct['x']),0)
		y = torch.unsqueeze(torch.from_numpy(struct['x_blind']),0)
		sample = [x, y]
		return sample



class LevinData(Dataset):
	def __init__(self, ALPHA):
		self.list_files = []
		self.OUT_DIR = './experiments/exp5/alpha'+str(int(ALPHA))+'/'
		for IM_IDX in range(1,5):
			for K_IDX in range(1,9):
				self.list_files.append(self.OUT_DIR+'no_em'+str(IM_IDX)+'_'+str(K_IDX)+'.mat')

	def __len__(self):
		return len(self.list_files)
		

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_file = self.list_files[idx]
		struct = loadmat(img_file)
		
		x = torch.unsqueeze(torch.from_numpy(struct['x']),0)
		y = torch.unsqueeze(torch.from_numpy(struct['x_blind']),0)
		sample = [x, y]
		return sample



class BSDData(Dataset):
	def __init__(self, ALPHA, N_FILES=100):
		self.list_files = []
		self.OUT_DIR = './experiments/exp4/alpha'+str(int(ALPHA))+'/'
		for IM_IDX in range(N_FILES):
			self.list_files.append(self.OUT_DIR+'blind'+str(IM_IDX)+'.mat')

	def __len__(self):
		return len(self.list_files) 	

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		img_file = self.list_files[idx]
		struct = loadmat(img_file)
		
		x = torch.unsqueeze(torch.from_numpy(struct['x']),0)
		y = torch.unsqueeze(torch.from_numpy(struct['x_blind']),0)
		sample = [x, y]
		return sample