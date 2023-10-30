
import torchvision as tv
import torch
import numpy as np

if torch.cuda.is_available():     # Make sure GPU is available
    print("CUDA is ready! Let go DL")
    dev = torch.device("cuda:0")
    kwar = {'num_workers': 8, 'pin_memory': True}
    cpu = torch.device("cpu")
else:
    print("Warning: CUDA not found, CPU only.")
    dev = torch.device("cpu")
    kwar = {}
    cpu = torch.device("cpu")

np.random.seed(551)

toTensor = tv.transforms.ToTensor()


def scaleImage(x):          # Pass a PIL image, return a tensor
    y = toTensor(x)
    if(y.min() < y.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        y = (y - y.min())/(y.max() - y.min())
    z = y - y.mean()        # Subtract the mean value of the image
    return z


def scaleBack(x):  # Pass a tensor, return a numpy array from 0 to 1
    if (x.min() < x.max()):  # Assuming the image isn't empty, rescale so its values run from 0 to 1
        x = (x - x.min()) / (x.max() - x.min())
    return x[0].to(cpu).numpy()  # Remove channel (grayscale anyway)