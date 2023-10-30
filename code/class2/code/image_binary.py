import SimpleITK as sitk
img=sitk.ReadImage('./data/lung.nii.gz')
seg_img=sitk.BinaryThreshold(img,500,1000)      #分割骨头
seg_img=sitk.BinaryMorphologicalClosing(seg_img)
seg_img=sitk.BinaryMorphologicalClosing(seg_img)
seg_img=sitk.BinaryMorphologicalClosing(seg_img)
sitk.WriteImage(seg_img,'./result/bone.nii.gz')


# from skimage import  morphology
#
# from Niiplot import multi_slice_viewer
#
# img=sitk.ReadImage("../2/lung.nii.gz")
# print(img)
# # img=sitk.BinaryThreshold(img,-600,-400)
#
# img=sitk.BinaryThreshold(img,-900,-400)
#
# img=sitk.BinaryMorphologicalClosing(img)
# # img=sitk.BinaryFillhole(img)
# # img=sitk.ThresholdMaximumConnectedComponents(img,11)
# # img_arr=sitk.GetArrayFromImage(img)
# # img_arr=morphology.remove_small_holes(img_arr)
# #
# # new_img=sitk.GetImageFromArray(img)
# # new_img.CopyInformation(img)
# sitk.WriteImage(img,'../2/binary.nii.gz')
# # multi_slice_viewer(img)