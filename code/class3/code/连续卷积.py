import SimpleITK as sitk

img=sitk.ReadImage('data/lung.nii.gz')
img=sitk.Cast(img,sitk.sitkFloat32)
img=sitk.DiscreteGaussian(img,[3,3,3])

img=sitk.SobelEdgeDetection(img)
sitk.WriteImage(img,'result/lung_edege.nii.gz')