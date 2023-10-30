import SimpleITK as sitk
img=sitk.ReadImage("data/lung.nii.gz")
img=sitk.Cast(img,sitk.sitkFloat32)
img=sitk.SobelEdgeDetection(img)
sitk.WriteImage(img,'result/lung_filter.nii.gz')



# import SimpleITK as sitk
# from Niiplot import multi_slice_viewer
# img=sitk.ReadImage('lung.nii.gz')
# img=sitk.Cast(img,sitk.sitkFloat32)
# # edge=sitk.CannyEdgeDetection(img,lowerThreshold=200, upperThreshold=400)
# edge=sitk.SobelEdgeDetection(img)
# multi_slice_viewer(edge)
# sitk.WriteImage(edge, 'lung_edege.nii.gz')