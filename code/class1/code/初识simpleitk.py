import SimpleITK as sitk

img = sitk.ReadImage('test.nii.gz')
print(img)
print(f'size:{img.GetSize()}')
print(f'spacing:{img.GetSpacing()}')

# scalarImage = sitk.Cast(img,sitk.sitkUInt32)
# sitk.Show(scalarImage)