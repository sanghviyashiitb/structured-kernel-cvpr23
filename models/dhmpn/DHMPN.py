import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		#Conv1
		self.layer1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1)
			)
		self.layer3 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1)
			)
		#Conv2
		self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
		self.layer6 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1)
			)
		self.layer7 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1)
			)
		#Conv3
		self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
		self.layer10 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1)
			)
		self.layer11 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1)
			)
		
	def forward(self, x):
		#Conv1
		x = self.layer1(x)
		x = self.layer2(x) + x
		x = self.layer3(x) + x
		#Conv2
		x = self.layer5(x)
		x = self.layer6(x) + x
		x = self.layer7(x) + x
		#Conv3
		x = self.layer9(x)    
		x = self.layer10(x) + x
		x = self.layer11(x) + x 
		return x

class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()        
		# Deconv3
		self.layer13 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1)
			)
		self.layer14 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1)
			)
		self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
		#Deconv2
		self.layer17 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1)
			)
		self.layer18 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1)
			)
		self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
		#Deconv1
		self.layer21 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1)
			)
		self.layer22 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv2d(32, 32, kernel_size=3, padding=1)
			)
		self.layer24 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
		
	def forward(self,x):        
		#Deconv3
		x = self.layer13(x) + x
		x = self.layer14(x) + x
		x = self.layer16(x)                
		#Deconv2
		x = self.layer17(x) + x
		x = self.layer18(x) + x
		x = self.layer20(x)
		#Deconv1
		x = self.layer21(x) + x
		x = self.layer22(x) + x
		x = self.layer24(x)
		return x

class DHMPN(nn.Module):
	def __init__(self):
		super(DHMPN, self).__init__()        
		self.encoder_lv1 = Encoder()
		self.encoder_lv2 = Encoder()    
		self.encoder_lv3 = Encoder()
		self.encoder_lv4 = Encoder()

		self.decoder_lv1 = Decoder()
		self.decoder_lv2 = Decoder()    
		self.decoder_lv3 = Decoder()
		self.decoder_lv4 = Decoder()

	def load(self, ckpt_path):
		self.encoder_lv1.load_state_dict(torch.load(ckpt_path + 'encoder_lv1.pkl'))
		self.encoder_lv2.load_state_dict(torch.load(ckpt_path + 'encoder_lv2.pkl'))
		self.encoder_lv3.load_state_dict(torch.load(ckpt_path + 'encoder_lv3.pkl'))
		self.encoder_lv4.load_state_dict(torch.load(ckpt_path + 'encoder_lv4.pkl'))
	
		self.decoder_lv1.load_state_dict(torch.load(ckpt_path + 'decoder_lv1.pkl'))
		self.decoder_lv2.load_state_dict(torch.load(ckpt_path + 'decoder_lv2.pkl'))
		self.decoder_lv3.load_state_dict(torch.load(ckpt_path + 'decoder_lv3.pkl'))
		self.decoder_lv4.load_state_dict(torch.load(ckpt_path + 'decoder_lv4.pkl'))
	

	def forward(self, img):
		_, _, H, W = img.size()          
		images_lv1 = img
		images_lv2_1 = images_lv1[:,:,0:int(H/2),:]
		images_lv2_2 = images_lv1[:,:,int(H/2):H,:]
		images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
		images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
		images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
		images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]
		images_lv4_1 = images_lv3_1[:,:,0:int(H/4),:]
		images_lv4_2 = images_lv3_1[:,:,int(H/4):int(H/2),:]
		images_lv4_3 = images_lv3_2[:,:,0:int(H/4),:]
		images_lv4_4 = images_lv3_2[:,:,int(H/4):int(H/2),:]
		images_lv4_5 = images_lv3_3[:,:,0:int(H/4),:]
		images_lv4_6 = images_lv3_3[:,:,int(H/4):int(H/2),:]
		images_lv4_7 = images_lv3_4[:,:,0:int(H/4),:]
		images_lv4_8 = images_lv3_4[:,:,int(H/4):int(H/2),:]

		feature_lv4_1 = self.encoder_lv4(images_lv4_1)
		feature_lv4_2 = self.encoder_lv4(images_lv4_2)
		feature_lv4_3 = self.encoder_lv4(images_lv4_3)
		feature_lv4_4 = self.encoder_lv4(images_lv4_4)
		feature_lv4_5 = self.encoder_lv4(images_lv4_5)
		feature_lv4_6 = self.encoder_lv4(images_lv4_6)
		feature_lv4_7 = self.encoder_lv4(images_lv4_7)
		feature_lv4_8 = self.encoder_lv4(images_lv4_8)
		
		feature_lv4_top_left = torch.cat((feature_lv4_1, feature_lv4_2), 2)
		feature_lv4_top_right = torch.cat((feature_lv4_3, feature_lv4_4), 2)
		feature_lv4_bot_left = torch.cat((feature_lv4_5, feature_lv4_6), 2)
		feature_lv4_bot_right = torch.cat((feature_lv4_7, feature_lv4_8), 2)
		feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
		feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
		feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)

		residual_lv4_top_left = self.decoder_lv4(feature_lv4_top_left)
		residual_lv4_top_right = self.decoder_lv4(feature_lv4_top_right)
		residual_lv4_bot_left = self.decoder_lv4(feature_lv4_bot_left)
		residual_lv4_bot_right = self.decoder_lv4(feature_lv4_bot_right)

		feature_lv3_1 = self.encoder_lv3(images_lv3_1 + residual_lv4_top_left)
		feature_lv3_2 = self.encoder_lv3(images_lv3_2 + residual_lv4_top_right)
		feature_lv3_3 = self.encoder_lv3(images_lv3_3 + residual_lv4_bot_left)
		feature_lv3_4 = self.encoder_lv3(images_lv3_4 + residual_lv4_bot_right)
		feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3) + feature_lv4_top
		feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3) + feature_lv4_bot
		feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
		residual_lv3_top = self.decoder_lv3(feature_lv3_top)
		residual_lv3_bot = self.decoder_lv3(feature_lv3_bot)

		feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
		feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
		feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
		residual_lv2 = self.decoder_lv2(feature_lv2)

		feature_lv1 = self.encoder_lv1(images_lv1 + residual_lv2) + feature_lv2
		out = self.decoder_lv1(feature_lv1)
		
		return out