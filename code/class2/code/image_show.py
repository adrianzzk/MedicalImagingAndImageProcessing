import matplotlib.pyplot as plt
import SimpleITK as sitk

from Niiplot import multi_slice_viewer

img=sitk.ReadImage("../2/BG0001.nii.gz")
print(img)

img_arr=sitk.GetArrayFromImage(img)
img_slice=img_arr[100,...]
plt.imshow(img_slice,cmap='gray')
plt.waitforbuttonpress()

#multi_slice_viewer(img)




