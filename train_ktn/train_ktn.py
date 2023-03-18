import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from kernel_dataset import Kernels_As_ControlPoints

from torch.utils.data import Dataset, DataLoader
from models.ktn.kernel_mlp import Kernel_MLP

LEARNING_RATE = 1e-5
NUM_EPOCHS = 100
BATCH_SIZE = 4
CONTROL_POINTS = 8


"""
Initiate a model, and transfer to gpu
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Kernel_MLP(n_control_points=CONTROL_POINTS); model.to(device)
print("Number of GPUS available: ", torch.cuda.device_count())

"""
Transform image and blur operations
"""
data_train = Kernels_As_ControlPoints(train=True, n_control_points = CONTROL_POINTS)
train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
data_val = Kernels_As_ControlPoints(train=False, n_control_points = CONTROL_POINTS)
val_loader = DataLoader(data_val, batch_size= BATCH_SIZE, shuffle=False, num_workers=0)

"""
Setting up training with :
	1. L1 Loss
	2. AdamOptimizer 
"""
criterion_list  = [torch.nn.MSELoss()]
wt_list = [1.0]
criterion_l2  = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience= 10)

"""
Training starts here
"""
for epoch in range(NUM_EPOCHS):
	epoch_loss = 0
	model.train()
	"""
	Training Epoch
	"""
	with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', unit='its') as pbar:	
		for i, data in enumerate(train_loader):
			"""
			Get training data - [true image, noisy_blurred, kernel, photon level]
			"""
			control_vec, kernel = data
			control_vec, kernel = control_vec.to(device), kernel.to(device)					
			"""
			Forward Pass => calculating loss => computing gradients => Adam optimizer update
			"""
			# Forward Pass
			optimizer.zero_grad()
			out = model(control_vec)
			# Calculating training loss
			loss = 0
			for idx in range(len(wt_list)):
				loss += wt_list[idx]*criterion_list[idx](out.float(), kernel.float())
			# Backprop 
			loss.backward()
			# Adam optimizer step
			optimizer.step()

			epoch_loss += loss.item()
			pbar.update(1)
			pbar.set_postfix(**{'loss (batch)': loss.item()})
			if i > len(train_loader):
				break

	epoch_loss = epoch_loss/len(train_loader)
	print('Epoch: {}, Training Loss: {}, Current Learning Rate: {}'.format(epoch+1,epoch_loss,LEARNING_RATE))

	"""
	Validation Epoch
	"""
	val_loss, mse = 0, 0
	model.eval()
	with torch.no_grad(): # Don't maintain computation graph since no backprop reqd., saves GPU memory
		for i, data in enumerate(val_loader):
			"""
			Getting validation pair
			"""
			control_vec, kernel = data
			control_vec, kernel = control_vec.to(device), kernel.to(device)
				
			"""
			Forward Pass
			"""
			out = model(control_vec)
			
			"""
			Calculating L2 loss and training loss on the validation set
			"""
			loss = 0
			for idx in range(len(wt_list)):
				loss += wt_list[idx]*criterion_list[idx](out.float(), kernel.float())
			loss_l2 = criterion_l2(out, kernel)
			val_loss += loss.item()
			mse += loss_l2.item()
			if i > len(val_loader):
				break
	val_loss = val_loss/len(val_loader)
	mse = mse/len(val_loader)
	# scheduler.step(val_loss)
	


	"""
	Writing the epoch loss, validation loss to tensorboard for visualization
	"""
	print('Validation Loss, averaged per pixel: %0.6f'%(val_loss*4096))	
	for param_group in optimizer.param_groups:
		LEARNING_RATE = param_group['lr']
	torch.save(model.state_dict(), 'kernel_mlp'+str(CONTROL_POINTS)+'_latest.pth')
	if LEARNING_RATE < 1e-6:
		break
