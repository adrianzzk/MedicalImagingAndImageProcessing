import os
import argparse
import random
import shutil
from shutil import copyfile
from misc import printProgressBar
import glob
import os
import SimpleITK as sitk
import numpy as np
import cv2

def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)


def save_img_to_nii(data, path):
    rescaled_data = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
    sitk.WriteImage(sitk.GetImageFromArray(rescaled_data),path)
    # cv2.imwrite(path, rescaled_data)


def save_lab_to_nii(original_array, path, mapping_dict):
    # 创建一个映射字典，将原始值映射到目标值
    #mapping_dict = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}

    # 使用NumPy的向量化操作将原始数组映射到目标数组
    print(np.unique(original_array))
    target_array = np.vectorize(mapping_dict.get)(original_array)
    print(np.unique(target_array))
    # cv2.imwrite(path, target_array)
    sitk.WriteImage(sitk.GetImageFromArray(target_array), path)

def split_lab(img,outdir,name,map_dic):
    outs=[]
    img_arr=sitk.GetArrayFromImage(img)
    for i in range(img_arr.shape[0]):

        slice_arr=img_arr[i]
        tmp_name=f"{outdir}/{name}_{i:04}.nii.gz"
        print(tmp_name)
        save_lab_to_nii(slice_arr, tmp_name, map_dic)
        outs.append(tmp_name)
    return outs

def split_img(img,outdir,name):
    outs=[]
    img_arr=sitk.GetArrayFromImage(img)
    for i in range(img_arr.shape[0]):
        slice_arr=img_arr[i]
        tmp_name=f"{outdir}/{name}_{i:04}.nii.gz"
        save_img_to_nii(slice_arr, tmp_name)
        outs.append(tmp_name)
    return outs



def main(config):

    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)
    rm_mkdir(config.test_path)
    rm_mkdir(config.test_GT_path)
    rm_mkdir(config.tmp_path)

    # filenames = os.listdir(config.origin_data_path)
    imgs=glob.glob(f"{config.origin_data_path}/*_cine.nii.gz")
    labs=glob.glob(f"{config.origin_data_path}/*_cine_myops.nii.gz")

    data_list = []
    GT_list = []

    for img_path in imgs:
        img=sitk.ReadImage(img_path)[...,0]
        outs=split_img(img,config.tmp_path,f"{get_file_name(img_path)}_img")
        data_list.extend(outs)
    for lab_path in labs:
        print(lab_path)
        img=sitk.ReadImage(lab_path)
        outs=split_lab(img,config.tmp_path,f"{get_file_name(lab_path)}_lab",map_dic={0:0,4:0,6:0, 2221:1,1220:1,200:1, 500:1, 600:0})
        GT_list.extend(outs)

    num_total = len(data_list)
    num_train = int((config.train_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_valid = int((config.valid_ratio/(config.train_ratio+config.valid_ratio+config.test_ratio))*num_total)
    num_test = num_total - num_train - num_valid

    print('\nNum of train set : ',num_train)
    print('\nNum of valid set : ',num_valid)
    print('\nNum of test set : ',num_test)

    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train):
        move_to_dst(data_list, GT_list, config.train_path, config.train_GT_path, Arange)
        printProgressBar(i + 1, num_train, prefix = 'Producing train set:', suffix = 'Complete', length = 50)
        

    for i in range(num_valid):
        move_to_dst(data_list, GT_list, config.valid_path, config.valid_GT_path, Arange)

        printProgressBar(i + 1, num_valid, prefix = 'Producing valid set:', suffix = 'Complete', length = 50)

    for i in range(num_test):
        move_to_dst(data_list, GT_list, config.test_path, config.test_GT_path, Arange)



        printProgressBar(i + 1, num_test, prefix = 'Producing test set:', suffix = 'Complete', length = 50)

def get_file_name(path):
    file_path=os.path.basename(path)
    file_name, file_extension = os.path.splitext(file_path)
    return file_name

def move_to_dst(data_list, GT_list, valid_path, valid_GT_path, Arange):
    idx = Arange.pop()
    print(f"moving {idx}")
    src = data_list[idx]
    dst = os.path.join(valid_path, os.path.basename(data_list[idx]))
    copyfile(src, dst)
    src = GT_list[idx]
    dst = os.path.join(valid_GT_path, os.path.basename(GT_list[idx]))
    copyfile(src, dst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)

    # data path
    parser.add_argument('--origin_data_path', type=str, default='L:/workspace/homework5/cine-seg/cine_data_fudan/cinemyops-50cases4dwb_rj_zxh')
    # parser.add_argument('--origin_GT_path', type=str, default='../ISIC/dataset/ISIC2018_Task1_Training_GroundTruth')
    
    parser.add_argument('--train_path', type=str, default='./dataset_nii/train/')
    parser.add_argument('--train_GT_path', type=str, default='./dataset_nii/train_GT/')
    parser.add_argument('--valid_path', type=str, default='./dataset_nii/valid/')
    parser.add_argument('--valid_GT_path', type=str, default='./dataset_nii/valid_GT/')
    parser.add_argument('--test_path', type=str, default='./dataset_nii/test/')
    parser.add_argument('--test_GT_path', type=str, default='./dataset_nii/test_GT/')
    parser.add_argument('--tmp_path', type=str, default='./dataset_nii/tmp/')

    config = parser.parse_args()
    print(config)
    main(config)