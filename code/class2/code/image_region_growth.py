import SimpleITK as sitk
from skimage import  morphology

from Niiplot import multi_slice_viewer

img=sitk.ReadImage("./data/lung.nii.gz")
print(img)
# img=sitk.BinaryThreshold(img,-600,-400)
slt=[]
slt.append([332,295,52])
slt.append([121,326,37])
img=sitk.ConnectedThreshold(img,slt,-900,-400)
img=sitk.BinaryMorphologicalClosing(img)
# img=sitk.BinaryFillhole(img)
# img=sitk.ThresholdMaximumConnectedComponents(img,11)
# img_arr=sitk.GetArrayFromImage(img)
# img_arr=morphology.remove_small_holes(img_arr)
# new_img=sitk.GetImageFromArray(img)
# new_img.CopyInformation(img)
sitk.WriteImage(img,'./result/binary.nii.gz')
# multi_slice_viewer(img)