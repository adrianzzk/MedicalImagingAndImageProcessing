import SimpleITK as sitk
# pip install SimpleITK -i https://mirrors.aliyun.com/pypi/simple
img=sitk.ReadImage("../output_1.2.840.113619.2.80.2818047608.9510.1689145036.1.4.1.nii.gz")
print(img)


