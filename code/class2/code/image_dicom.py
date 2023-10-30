import SimpleITK as sitk
print(f'version: {sitk.__version__}')
reader=sitk.ImageSeriesReader()
ids=reader.GetGDCMSeriesIDs('../data/23071209/23150000/')
for id in ids:
    names=reader.GetGDCMSeriesFileNames('../data/23071209/23150000/',id)
    reader.SetFileNames(names)
    img=reader.Execute()
    print(img)
    sitk.WriteImage(img,f'../output_{id}.nii.gz')
    print('finished')


