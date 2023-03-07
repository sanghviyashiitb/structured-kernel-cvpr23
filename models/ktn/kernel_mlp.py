import torch
import torch.nn as nn
import models.resunet.resnet_basicblock as B
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Kernel_MLP(nn.Module):
	def __init__(self, n_control_points = 3, ngf = 32, k_size = 64):

		super(Kernel_MLP, self).__init__()
		self.input_dim = 2*(n_control_points-1)

		self.k_size = k_size
		self.ngf = ngf
		self.mlp = nn.Sequential(
			nn.Linear(self.input_dim, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, self.ngf*32),
			nn.ReLU(inplace=True),
			nn.Linear(self.ngf*32, self.ngf*32)
		)
		upsample_block = B.upsample_convtranspose; nb = 2; act_mode = 'R'
		# Input N X ngf*32 X 1 X 1
		self.m_up5 = B.sequential(upsample_block(ngf*32, ngf*16, bias=False, mode='2'), *[B.ResBlock(ngf*16, ngf*16, bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
		# Input N X ngf*16 X 2 X 2
		self.m_up4 = B.sequential(upsample_block(ngf*16, ngf*8, bias=False, mode='2'), *[B.ResBlock(ngf*8, ngf*8, bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
		# Input N X ngf*8 X 4 X 4
		self.m_up3 = B.sequential(upsample_block(ngf*8, ngf*4, bias=False, mode='2'), *[B.ResBlock(ngf*4, ngf*4, bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
		# Input N X ngf*4 X 8 X 8
		self.m_up2 = B.sequential(upsample_block(ngf*4, ngf*2, bias=False, mode='2'), *[B.ResBlock(ngf*2, ngf*2, bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
		# Input N X ngf*2 X 16 X 16
		self.m_up1 = B.sequential(upsample_block(ngf*2, ngf*2, bias=False, mode='2'), *[B.ResBlock(ngf*2, ngf*2, bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
		# Input N X ngf X 32 X 32
		self.m_up0 = B.sequential(upsample_block(ngf*2, ngf*2, bias=False, mode='2'), *[B.ResBlock(ngf*2, ngf*2, bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
		# Output N X ngf X 64 X 64
		self.m_tail = B.conv(ngf*2, 1, bias=False, mode='C')
		self.relu_tail = nn.ReLU(inplace=True)

	def n_control_points(self):
		return self.input_dim//2 + 1

	def forward(self, input):
		N = input.size(0)
		k_feat  = self.mlp(input/64.0).view(N,self.ngf*32,1,1)
		k_out = self.m_up5(k_feat)
		k_out = self.m_up4(k_out)
		k_out = self.m_up3(k_out)
		k_out = self.m_up2(k_out)
		k_out = self.m_up1(k_out)
		k_out = self.m_up0(k_out)
		k_out = self.m_tail(k_out)
		if not self.training:
			k_out = self.relu_tail(k_out)
			k_sum = torch.sum(k_out, dim=[1,2,3], keepdim=True)
			k_out = k_out/k_sum
		return  k_out
