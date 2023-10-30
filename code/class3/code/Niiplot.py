import numpy as np
import matplotlib.pyplot as plt

import SimpleITK as sitk
import skimage.io as io
import pylab
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
def read_img(path):
    img=sitk.ReadImage(path)
    print(img)
    data=sitk.GetArrayFromImage(img)
    return data
def show_img(data):
    for i in range(data.shape[0]):
        io.imshow(data[i,:,:],cmap="gray")
        print(i)
        io.show()
def show_slice(data):
    io.imshow(data, cmap="gray")
    io.show()

def multi_slice_viewer(volume):
    if isinstance(volume,sitk.Image):
        volume=sitk.GetArrayViewFromImage(volume)
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume

    ax.index = volume.shape[0] // 2

    ax.imshow(volume[ax.index],cmap = 'gray')

    fig.canvas.mpl_connect('key_press_event', process_key)
    pylab.show()
def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()
def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    ax.set_title(ax.index)
def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)
def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    ax.set_title(ax.index)





