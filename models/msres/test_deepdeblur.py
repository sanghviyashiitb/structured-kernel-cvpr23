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
from utils.dataloader import Poiss_List, GoPro

from models.deep_deblur.MSResNet import MSResNet


BATCH_SIZE=256

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MSResNet()

model.load_state_dict(torch.load('/scratch/gilbreth/mao114/model_zoo/deepdeblur_gopro.pth'))

transform_noise = Poiss_List([1,60])
data_val = GoPro(False,  transform_noise)

val_loader = DataLoader(data_val, batch_size=BATCH_SIZE, shuffle=False)
criterion_l2  = torch.nn.MSELoss()


model.eval()
model = model.to(device)

torch.set_grad_enabled(False)

mse=0

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
    out = model(y_M)[0]
            
    loss_l2 = criterion_l2(out, x)
    mse += loss_l2.item()
mse = mse*BATCH_SIZE*4/len(data_val)
psnr = -10*np.log10(mse)

print(psnr)
        
        
        
        
        
        
        

