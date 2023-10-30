import os

from PIL.Image import Image
from torch import optim

import torch.nn.functional as F
from torchvision.transforms import transforms

from dataset import rm_mkdir
from evaluation import *
from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net
import csv
import torch.nn as nn
from tensorboardX import SummaryWriter

class Experiment(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader
		self.config=config
		# Models
		self.unet = None
		self.optimizer = None
		self.img_ch = config.img_ch
		self.output_ch = config.output_ch
		self.criterion = torch.nn.BCELoss()
		self.augmentation_prob = config.augmentation_prob

		# Hyper-parameters
		self.lr = config.lr
		self.beta1 = config.beta1
		self.beta2 = config.beta2

		# Training settings
		self.num_epochs = config.num_epochs
		self.num_epochs_decay = config.num_epochs_decay
		self.batch_size = config.batch_size

		# Step size
		self.log_step = config.log_step
		self.val_step = config.val_step
		rm_mkdir(f"{config.result_path}/logs")
		self.write=SummaryWriter(f"{config.result_path}/logs")
		# Path
		self.model_path = config.model_path
		self.result_path = config.result_path
		self.mode = config.mode

		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model_type = config.model_type
		self.t = config.t
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		if self.model_type =='U_Net':
			self.unet = U_Net(img_ch=self.config.img_ch,output_ch=self.config.output_ch)
		elif self.model_type =='R2U_Net':
			self.unet = R2U_Net(img_ch=self.config.img_ch,output_ch=self.config.output_ch,t=self.t)
		elif self.model_type =='AttU_Net':
			self.unet = AttU_Net(img_ch=self.config.img_ch,output_ch=self.config.output_ch)
		elif self.model_type == 'R2AttU_Net':
			self.unet = R2AttU_Net(img_ch=self.config.img_ch,output_ch=self.config.output_ch,t=self.t)
			

		self.optimizer = optim.Adam(list(self.unet.parameters()),
									  self.lr, [self.beta1, self.beta2])
		self.unet.to(self.device)

		# self.print_network(self.unet, self.model_type)

	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def to_data(self, x):
		"""Convert variable to tensor."""
		if torch.cuda.is_available():
			x = x.cpu()
		return x.data

	def update_lr(self, g_lr, d_lr):
		for param_group in self.optimizer.param_groups:
			param_group['lr'] = self.config.lr

	def reset_grad(self):
		"""Zero the gradient buffers."""
		self.unet.zero_grad()

	def compute_accuracy(self,SR,GT):
		SR_flat = SR.view(-1)
		GT_flat = GT.view(-1)

		acc = GT_flat.data.cpu()==(SR_flat.data.cpu()>0.5)

	def tensor2img(self,x):
		img = (x[:,0,:,:]>x[:,1,:,:]).float()
		img = img*255
		return img


	# 定义Dice损失函数
	def dice_loss(self,predicted_one_hot, true_one_hot, smooth=1e-5):
		# intersection = (pred * target).sum()
		# union = pred.sum() + target.sum() + smooth
		# return 1 - (2.0 * intersection + smooth) / union

		intersection = torch.sum(predicted_one_hot * true_one_hot, dim=(2, 3))
		union = torch.sum(predicted_one_hot, dim=(2, 3)) + torch.sum(true_one_hot, dim=(2, 3))
		dice_scores = (2.0 * intersection + smooth) / (union + smooth)  # 添加平滑项以避免除零错误
		dice_loss = 1 - dice_scores.mean()
		return dice_loss

	def conve_one_hot(self,label_image,num_classes=2):

		label_image=label_image.to(torch.uint8)
		# 对标签进行one-hot编码
		one_hot_label = torch.zeros(label_image.size(0),num_classes, label_image.size(2), label_image.size(3))
		for i in range(num_classes):
			one_hot_label[:,i,...] = (label_image == i).float()

		return one_hot_label

	def test(self,unet_path):


		# ===================================== Test ====================================#

		self.build_model()
		self.unet.load_state_dict(torch.load(unet_path))

		self.unet.train(False)
		self.unet.eval()

		for i, (images, GT) in enumerate(self.valid_loader):
			images = images.to(self.device)
			GT = GT.to(self.device)
			SR = self.unet(images)

			for t in range(images.shape[0]):
				image = images[t].detach()  # 获取一张图像
				segmentation = GT[t].detach()  # 获取对应的分割结果
				output = torch.argmax(SR[t], dim=0, keepdim=True).detach()

				self.write.add_image(f"Image", image, global_step=i)
				self.write.add_image(f"Output", output, global_step=i)
				self.write.add_image(f"GT", segmentation, global_step=i)
				self.write.flush()

	def train(self):
		"""Train encoder, generator and discriminator."""

		#====================================== Training ===========================================#
		#===========================================================================================#

		unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' %(self.model_type,self.num_epochs,self.lr,self.num_epochs_decay,self.augmentation_prob))

		# U-Net Train
		if os.path.isfile(unet_path):
			# Load the pretrained Encoder
			self.unet.load_state_dict(torch.load(unet_path))
			print('%s is Successfully Loaded from %s'%(self.model_type,unet_path))
		else:
			# Train for Encoder
			lr = self.lr
			best_unet_score = 0.
			
			for epoch in range(self.num_epochs):

				self.unet.train(True)
				epoch_loss = 0
				

				criterion_ce = nn.CrossEntropyLoss()
				for i, (images, GT) in enumerate(self.train_loader):
					# GT : Ground Truth

					images = images.to(self.device)
					GT = GT.to(self.device)
					SR = self.unet(images)

					# 计算交叉熵损失和Dice损失
					ce_loss = criterion_ce(SR, GT.squeeze())
					dice = self.dice_loss(torch.softmax(SR, dim=1), F.one_hot(GT.squeeze(), 2).permute(0, 3, 1, 2).to(torch.float32))
					loss=ce_loss+dice

					epoch_loss += loss.item()
					print(f"ce loss:{ce_loss.item()} dice loss:{dice.item()}")
					# Backprop + optimize
					self.reset_grad()
					loss.backward()
					self.optimizer.step()

					for t in range(images.shape[0]):
						image = images[t].detach()  # 获取一张图像
						segmentation = GT[t].detach() # 获取对应的分割结果
						output=torch.argmax(SR[t],dim=0,keepdim=True).detach()
						self.write.add_scalar(f"ce_loss",ce_loss.item(), global_step=epoch*len(self.train_loader)+i)
						self.write.add_scalar(f"dice_loss",dice.item(), global_step=epoch*len(self.train_loader)+i)
						self.write.add_image(f"Image", image, global_step=epoch*len(self.train_loader)+i)
						self.write.add_image(f"Output", output, global_step=epoch*len(self.train_loader)+i)
						self.write.add_image(f"GT", segmentation, global_step=epoch*len(self.train_loader)+i)
						self.write.flush()


				# Print the log info
				print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, self.num_epochs,  epoch_loss))
				# ===================================== Validation ====================================#
				self.unet.train(False)
				self.unet.eval()

				best_unet = self.unet.state_dict()
				print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
				torch.save(best_unet, unet_path)

			self.write.close()



			
