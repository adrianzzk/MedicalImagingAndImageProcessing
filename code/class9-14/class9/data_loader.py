import os
import random
from random import shuffle
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
from pre_processing import *

class ImageFolder(data.Dataset):
	def __init__(self, img_root, image_size=224, mode='train', augmentation_prob=0.4,in_size=256, out_size=256):
		"""Initializes image paths and preprocessing module."""
		# self.root = img_root
		
		# GT : Ground Truth
		lab_root = img_root[:-1] + '_GT/'

		self.image_paths = list(map(lambda x: os.path.join(img_root, x), os.listdir(img_root)))
		self.label_paths = list(map(lambda x: os.path.join(lab_root, x), os.listdir(lab_root)))
		self.image_size = image_size
		self.in_size, self.out_size = in_size, out_size
		self.mode = mode
		self.RotationDegree = [0,90,180,270]
		self.augmentation_prob = augmentation_prob
		print("image count in {} path :{}".format(self.mode,len(self.image_paths)))

	def __getitem__(self, index):
		"""Get specific data corresponding to the index
        Args:
            index (int): index of the data
        Returns:
            Tensor: specific data on index which is converted to Tensor
        """
		"""
        # GET IMAGE
        """
		single_image_name = self.image_paths[index]
		img_as_img = Image.open(single_image_name)
		# img_as_img.show()
		img_as_np = np.asarray(img_as_img)

		# Augmentation
		# flip {0: vertical, 1: horizontal, 2: both, 3: none}
		flip_num = randint(0, 3)
		img_as_np = flip(img_as_np, flip_num)

		# Noise Determine {0: Gaussian_noise, 1: uniform_noise
		if randint(0, 1):
			# Gaussian_noise
			gaus_sd, gaus_mean = randint(0, 20), 0
			img_as_np = add_gaussian_noise(img_as_np, gaus_mean, gaus_sd)
		else:
			# uniform_noise
			l_bound, u_bound = randint(-20, 0), randint(0, 20)
			img_as_np = add_uniform_noise(img_as_np, l_bound, u_bound)

		# Brightness
		pix_add = randint(-20, 20)
		img_as_np = change_brightness(img_as_np, pix_add)

		# Elastic distort {0: distort, 1:no distort}
		sigma = randint(6, 12)
		# sigma = 4, alpha = 34
		img_as_np, seed = add_elastic_transform(img_as_np, alpha=34, sigma=sigma, pad_size=20)

		# Crop the image
		img_height, img_width = img_as_np.shape[0], img_as_np.shape[1]
		pad_size = int((self.in_size - self.out_size) / 2)
		img_as_np = np.pad(img_as_np, pad_size, mode="symmetric")
		y_loc, x_loc = randint(0, img_height - self.out_size), randint(0, img_width - self.out_size)
		img_as_np = cropping(img_as_np, crop_size=self.in_size, dim1=y_loc, dim2=x_loc)
		'''
        # Sanity Check for image
        img1 = Image.fromarray(img_as_np)
        img1.show()
        '''
		# Normalize the image
		img_as_np = normalization2(img_as_np, max=1, min=0)
		img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
		img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

		"""
        # GET MASK
        """
		single_mask_name = self.label_paths[index]
		msk_as_img = Image.open(single_mask_name)
		# msk_as_img.show()
		msk_as_np = np.asarray(msk_as_img)

		# flip the mask with respect to image
		msk_as_np = flip(msk_as_np, flip_num)

		# elastic_transform of mask with respect to image

		# sigma = 4, alpha = 34, seed = from image transformation
		msk_as_np, _ = add_elastic_transform(
			msk_as_np, alpha=34, sigma=sigma, seed=seed, pad_size=20)
		# msk_as_np = approximate_image(msk_as_np)  # images only with 0 and 255

		# Crop the mask
		msk_as_np = cropping(msk_as_np, crop_size=self.out_size, dim1=y_loc, dim2=x_loc)
		'''
        # Sanity Check for mask
        img2 = Image.fromarray(msk_as_np)
        img2.show()
        '''

		# Normalize mask to only 0 and 1
		# msk_as_np = msk_as_np / 255
		# msk_as_np = np.expand_dims(msk_as_np, axis=0)  # add additional dimension
		msk_as_tensor = torch.from_numpy(msk_as_np).long()  # Convert numpy array to tensor
		msk_as_tensor=msk_as_tensor.unsqueeze(0)
		return (img_as_tensor, msk_as_tensor)

	# def __getitem__(self, index):
	# 	"""Reads an image from a file and preprocesses it and returns."""
	# 	image_path = self.image_paths[index]
	# 	# filename = image_path.split('_')[-1][:-len(".jpg")]
	# 	# GT_path = self.GT_paths + 'ISIC_' + filename + '_segmentation.png'
	# 	GT_path=self.label_paths[index]
	#
	# 	# print(f"{image_path}  {GT_path}")
	#
	# 	image = Image.open(image_path)
	# 	GT = Image.open(GT_path)
	#
	# 	aspect_ratio = image.size[1]/image.size[0]
	#
	# 	Transform = []
	#
	# 	ResizeRange = random.randint(300,320)
	# 	Transform.append(T.Resize((int(ResizeRange*aspect_ratio),ResizeRange)))
	# 	p_transform = random.random()
	#
	# 	if (self.mode == 'train') and p_transform <= self.augmentation_prob:
	# 		RotationDegree = random.randint(0,3)
	# 		RotationDegree = self.RotationDegree[RotationDegree]
	# 		if (RotationDegree == 90) or (RotationDegree == 270):
	# 			aspect_ratio = 1/aspect_ratio
	#
	# 		Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
	#
	# 		RotationRange = random.randint(-10,10)
	# 		Transform.append(T.RandomRotation((RotationRange,RotationRange)))
	# 		CropRange = random.randint(250,270)
	# 		Transform.append(T.CenterCrop((int(CropRange*aspect_ratio),CropRange)))
	# 		Transform = T.Compose(Transform)
	#
	# 		image = Transform(image)
	# 		GT = Transform(GT)
	#
	# 		ShiftRange_left = random.randint(0,20)
	# 		ShiftRange_upper = random.randint(0,20)
	# 		ShiftRange_right = image.size[0] - random.randint(0,20)
	# 		ShiftRange_lower = image.size[1] - random.randint(0,20)
	# 		image = image.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
	# 		GT = GT.crop(box=(ShiftRange_left,ShiftRange_upper,ShiftRange_right,ShiftRange_lower))
	#
	# 		if random.random() < 0.5:
	# 			image = F.hflip(image)
	# 			GT = F.hflip(GT)
	#
	# 		if random.random() < 0.5:
	# 			image = F.vflip(image)
	# 			GT = F.vflip(GT)
	#
	# 		# Transform = T.ColorJitter(brightness=0.02,contrast=0.02,hue=0.02)
	#
	# 		# image = Transform(image)
	#
	# 		Transform =[]
	#
	#
	# 	Transform.append(T.Resize((int(256*aspect_ratio)-int(256*aspect_ratio)%16,256)))
	# 	Transform.append(T.ToTensor())
	# 	Transform = T.Compose(Transform)
	#
	# 	image = Transform(image)
	# 	GT = Transform(GT)
	# 	GT=GT.to(torch.long)
	# 	# GT=self.conve_one_hot(GT)
	#
	# 	Norm_ = T.Normalize(0, 1)
	# 	image = Norm_(image)
	#
	# 	return image, GT
	#

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = ImageFolder(img_root= image_path, image_size =image_size, mode=mode, augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
