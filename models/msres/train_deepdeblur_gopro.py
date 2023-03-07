# Comment this once done with debugging
# import warnings
# warnings.filterwarnings("ignore")


import numpy as np
from numpy.fft import fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Resize, RandomCrop, Grayscale, ToTensor, RandomVerticalFlip
from torchvision.transforms import RandomResizedCrop, Grayscale, ToTensor, RandomVerticalFlip

from utils.utils_deblur import gauss_kernel
from utils.utils_torch import MultiScaleLoss, conv_fft_batch, psf_to_otf
from utils.dataloader import GoPro, Poiss_List
from utils.mprnet.losses import CharbonnierLoss, EdgeLoss

from models.deep_deblur.MSResNet import MSResNet

# parser = argparse.ArgumentParser(description='deblur arguments')
# parser.add_argument('--model', type=str, default='mprnet', help='Model to train')
# args = parser.parse_args()


MODEL_TRAIN = 'deepdeblur_gopro'
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
BATCH_SIZE = 16
N_TRAIN = 128 
N_VAL = 256



"""
Initiate a model, and transfer to gpu
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MSResNet()
model.load_state_dict(torch.load('/scratch/gilbreth/mao114/model_zoo/deepdeblur_temp/deepdeblur_gopro_120epoch.pth'))

# writer = SummaryWriter()	# Tensorboard Writer
model.to(device)
# print("Number of GPUS available: ", torch.cuda.device_count())

# model = nn.DataParallel(model)
# model.to(device)
"""
Transform image and blur operations
"""
transform_noise = Poiss_List([1,60])

# Dataloaders
data_train = GoPro(True, transform_noise)
data_val = GoPro(False,  transform_noise)
train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
val_loader = DataLoader(data_val, batch_size=BATCH_SIZE*4, shuffle=False, num_workers=0)

criterion = torch.nn.L1Loss()
criterion_l2  = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE,  betas=(0.9, 0.999),eps=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=120)


""" 
Scheduler 
"""
# scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)

"""
Training starts here
"""
for epoch in range(NUM_EPOCHS):
		
	epoch_loss = 0
	model.train()

	"""
	Training Epoch
	"""
	with tqdm(total=len(data_train), desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', unit='its') as pbar:
		for i, data in enumerate(train_loader):
			"""
			Get training data - [true image, noisy_blurred, kernel, photon level]
			"""
			x, y, M = data
			M = M.view( x.size(0), 1, 1, 1)
			x, y_M = x.to(device), (y/M).float().to(device)
			 	
			"""
			Forward Pass => calculating loss => computing gradients => Adam optimizer update
			"""
			optimizer.zero_grad()
			# Forward Pass
			out= model(y_M)

			# Calculating training loss
            
			x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
			x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
			loss1 = criterion(out[0],x)
			loss2 = criterion(out[1],x_2)
			loss3 = criterion(out[2],x_4)
			loss = loss1+loss2+loss3

        # Backprop 
			loss.backward()

			# Adam optimizer step
			optimizer.step()

			epoch_loss += loss.item()
			pbar.update(BATCH_SIZE)
			pbar.set_postfix(**{'loss (batch)': loss.item()})
	epoch_loss = epoch_loss*BATCH_SIZE/len(data_train)
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
			x, y, M = data
			M = M.view( x.size(0), 1, 1, 1)
			x, y_M = x.to(device), (y/M).float().to(device)
			
			"""
			Forward Pass
			"""
			out = model(y_M)
			
			"""
			Calculating L2 loss and training loss on the validation set
			"""
			x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
			x_4 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
			loss1 = criterion(out[0],x)
			loss2 = criterion(out[1],x_2)
			loss3 = criterion(out[2],x_4)
			loss = loss1+loss2+loss3
        
			loss_l2 = criterion_l2(out[0], x)
			val_loss += loss.item()
			mse += loss_l2.item()

	val_loss = val_loss*BATCH_SIZE/len(data_val)
	mse = mse*BATCH_SIZE*4/len(data_val)
	psnr = -10*np.log10(mse)
	

	scheduler.step()
	"""
	Writing the epoch loss, validation loss to tensorboard for visualization
	"""
	print('Validation PSNR: %0.3f, Validation Loss: %0.6f'%(psnr, val_loss))	
	# writer.add_scalar("Loss/train", epoch_loss, epoch)
	# writer.add_scalar("Loss/learning_rate", LEARNING_RATE, epoch)
	# writer.add_scalar("Loss/val", val_loss, epoch)
	# writer.add_scalar("Loss/psnr", psnr, epoch)
	# for param_group in optimizer.param_groups:
		# LEARNING_RATE = param_group['lr']
	if epoch % 10 ==0:
		torch.save(model.state_dict(), '/scratch/gilbreth/mao114/model_zoo/'+MODEL_TRAIN+'_%depoch.pth'%(epoch))
# writer.close()




