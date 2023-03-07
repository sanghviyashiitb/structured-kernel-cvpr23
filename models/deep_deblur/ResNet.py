import torch.nn as nn

from models.deep_deblur import common


class ResNet(nn.Module):
	def __init__(self, in_channels=1, out_channels=1):
		super(ResNet, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.n_feats = 64
		self.kernel_size = 5
		self.n_resblocks = 19

		
		modules = []
		modules.append(common.default_conv(self.in_channels, self.n_feats, self.kernel_size))
		for _ in range(self.n_resblocks):
			modules.append(common.ResBlock(self.n_feats, self.kernel_size))
		modules.append(common.default_conv(self.n_feats, self.out_channels, self.kernel_size))

		self.body = nn.Sequential(*modules)

	def forward(self, input):
		
		output = self.body(input)

		
		return output
