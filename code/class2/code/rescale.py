import SimpleITK as sitk

img=sitk.ReadImage("../output_1.2.840.113619.2.80.2818047608.9510.1689145036.1.4.1.nii.gz")
print(img)
img=sitk.RescaleIntensity(img,0,255)
sitk.WriteImage(img,'../rescale.nii.gz')