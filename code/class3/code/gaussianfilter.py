import SimpleITK as sitk

from Niiplot import multi_slice_viewer

img=sitk.ReadImage('class3/data/lung.nii.gz')
img=sitk.Cast(img,sitk.sitkFloat32)
# edge=sitk.CannyEdgeDetection(img,lowerThreshold=200, upperThreshold=400)
img=sitk.DiscreteGaussian(img,[3,3,3])
multi_slice_viewer(img)
sitk.WriteImage(img, 'class3/result/lung_gaussain.nii.gz')