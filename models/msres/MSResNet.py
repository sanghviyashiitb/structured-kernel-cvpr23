import torch
import torch.nn as nn
import torch.nn.functional as F
from models.deep_deblur import common
from models.deep_deblur.ResNet import ResNet


class conv_end(nn.Module):
	def __init__(self, in_channels=1, out_channels=1, kernel_size=5, ratio=2):
		super(conv_end, self).__init__()

		modules = [
			common.default_conv(in_channels, out_channels, kernel_size),
			nn.PixelShuffle(ratio)
		]

		self.uppath = nn.Sequential(*modules)

	def forward(self, x):
		return self.uppath(x)

class MSResNet(nn.Module):
	def __init__(self):
		super(MSResNet, self).__init__()

		self.n_resblocks = 19
		self.n_feats = 64
		self.kernel_size = 5

		self.n_scales = 3

		self.body_models = nn.ModuleList([
			ResNet(1, 1)
		])
		for _ in range(1, self.n_scales):
			self.body_models.insert(0, ResNet(2, 1))

		self.conv_end_models = nn.ModuleList([None])
		for _ in range(1, self.n_scales):
			self.conv_end_models += [conv_end(1, 4)]

	def forward(self, input):
		scales = range(self.n_scales-1, -1, -1)    # 0: fine, 2: coarse


		input_pyramid = []
		input_pyramid.append(input)
		for s in range(1,self.n_scales):
			factor = 2**s
			input_coarse = F.interpolate(input, (input.shape[-2]//factor, input.shape[-1]//factor), mode='bilinear')
			input_pyramid.append(input_coarse)
			

		# for s in scales:
		# 	input_pyramid[s] = input_pyramid[s] - self.mean

		output_pyramid = [None] * self.n_scales

		input_s = input_pyramid[-1]
		for s in scales:    # [2, 1, 0]
			output_pyramid[s] = self.body_models[s](input_s)
			if s > 0:
				up_feat = self.conv_end_models[s](output_pyramid[s])
				input_s = torch.cat((input_pyramid[s-1], up_feat), 1)

		# for s in scales:
			# output_pyramid[s] = output_pyramid[s] + self.mean

		return output_pyramid[0]