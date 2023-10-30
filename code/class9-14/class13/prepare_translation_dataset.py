from util.dir import sort_glob,get_name_wo_suffix,mkdir_if_not_exist
import SimpleITK as sitk
import os
import numpy as np
def my_read(r8):
    print(r8)
    r8_itk = sitk.ReadImage(r8)
    r8_arr=sitk.GetArrayFromImage(r8_itk)
    return r8_itk,r8_arr

def clipseScaleSitkImage(sitk_image,low=5, up=95):
    np_image = sitk.GetArrayFromImage(sitk_image)
    # threshold image between p10 and p98 then re-scale [0-255]
    p0 = np_image.min().astype('float')
    p10 = np.percentile(np_image, low)
    p99 = np.percentile(np_image, up)
    p100 = np_image.max().astype('float')
    # logger.info('p0 {} , p5 {} , p10 {} , p90 {} , p98 {} , p100 {}'.format(p0,p5,p10,p90,p98,p100))
    sitk_image = sitk.Threshold(sitk_image,
                                lower=p10,
                                upper=p100,
                                outsideValue=p10)
    sitk_image = sitk.Threshold(sitk_image,
                                lower=p0,
                                upper=p99,
                                outsideValue=p99)
    sitk_image = sitk.RescaleIntensity(sitk_image,
                                       outputMinimum=0,
                                       outputMaximum=255)
    return sitk_image

def sitkResize(image, new_size, interpolator):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    new_spacing = [sz * spc / nsz for nsz, sz, spc in zip(new_size, image.GetSize(), image.GetSpacing())]
    resample.SetOutputSpacing(new_spacing)
    orig_size = np.array(image.GetSize(), dtype=np.int32)
    orig_spacing = list(image.GetSpacing())
    new_size=[oz*os/nz for oz,os,nz in zip(orig_size,orig_spacing,new_spacing)]
    new_size = np.ceil(new_size).astype(np.int32)  # Image dimensions are in integers
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    newimage = resample.Execute(image)
    return newimage

def extract(outdir,r8,lower=0.3,upper=0.7):
    mkdir_if_not_exist(os.path.dirname(outdir))
    name=get_name_wo_suffix(r8)
    name=name.replace(" ","_")
    r8_itk, r8_arr = my_read(r8)
    for i in range(int(r8_itk.GetSize()[-1] * lower), int(r8_itk.GetSize()[-1] * upper)):
        tmp = r8_itk[...,i]
        tmp=sitkResize(tmp,[256,256],sitk.sitkLinear)
        tmp=clipseScaleSitkImage(tmp,5,99)
        sitk.WriteImage(tmp,f"{outdir}_{name}_{i}.nii.gz")

if __name__=="__main__":
    patiens=sort_glob('../DSC_DWI_SWI/*')
    for idx, p in enumerate(patiens):
        slice=-1
        for token in ['Mag_Images','Pha_Images','CBV_REG']:
            print(p)
            r8=sort_glob(f'{p}/*{token}*.nii')
            if slice==-1:
                slice=sitk.ReadImage(r8).GetSize()[-1]
            else:
                assert slice==sitk.ReadImage(r8).GetSize()[-1]
            assert len(r8)==1
            p_name=os.path.basename(p)
            extract(f'./data/{token}/case{idx:04}',r8[0])







