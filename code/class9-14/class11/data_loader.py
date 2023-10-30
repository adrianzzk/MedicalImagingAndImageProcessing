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
import SimpleITK as sitk

class ImageDataset(data.Dataset):
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
		# SimpleITK
		single_image_name = self.image_paths[index] 
		img_as_img = sitk.ReadImage(single_image_name)
		img_as_np = sitk.GetArrayFromImage(img_as_img)
		# img_as_img.show()
		# img_as_np = np.asarray(img_as_img)

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
		# img_as_np = zscore(img_as_np)
		img_as_np = np.expand_dims(img_as_np, axis=0)  # add additional dimension
		img_as_tensor = torch.from_numpy(img_as_np).float()  # Convert numpy array to tensor

		"""
        # GET MASK
        """
		single_mask_name = self.label_paths[index]
		msk_as_img = sitk.ReadImage(single_mask_name)
		# msk_as_img.show()
		msk_as_np = sitk.GetArrayFromImage(msk_as_img)

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

	def __len__(self):
		"""Returns the total number of font files."""
		return len(self.image_paths)

def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train',augmentation_prob=0.4):
	"""Builds and returns Dataloader."""
	
	dataset = ImageDataset(img_root= image_path, image_size =image_size, mode=mode, augmentation_prob=augmentation_prob)
	data_loader = data.DataLoader(dataset=dataset,
								  batch_size=batch_size,
								  shuffle=True,
								  num_workers=num_workers)
	return data_loader
