import SimpleITK as sitk
import numpy as np

img=sitk.ReadImage('../2/BG0001.nii.gz')
img_arr=sitk.GetArrayFromImage(img)
mn=img_arr.mean()
sd=img_arr.std()
normalized=(img_arr-mn)/sd
normalized=sitk.GetImageFromArray(normalized)
normalized.CopyInformation(img)
sitk.WriteImage(normalized,'../2/normalized.nii.gz')