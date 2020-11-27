import shutil
from os.path import join, isfile

import h5py
import os
import numpy as np
import scipy.io as scio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

from data_preprocess.sun_rgbd.utils import load_list


def load_seg_label(path_seg_label):
    with h5py.File(path_seg_label, 'r') as f:
        SUNRGBD2Dseg = f['SUNRGBD2Dseg']
        seglabel = SUNRGBD2Dseg['seglabel']
        seg_labels = [np.array(SUNRGBD2Dseg[idx[0]]).swapaxes(0, 1).astype(np.uint8) # [:, :, np.newaxis]
                      for idx in seglabel]
    return seg_labels


def load_label_name(path_label_name):
    data = scio.loadmat(path_label_name)
    return [name[0] for name in data['seg37list'][0]]


def combine_imgdep_hha(dir_sun, dir_sun_hha, dir_sun_xyz, dir_sun_target, sub_dir):
    dir_img = os.path.join(dir_sun, sub_dir, 'image')
    dir_target_img = os.path.join(dir_sun_target, 'data', sub_dir)
    if not os.path.exists(dir_target_img):
        os.makedirs(dir_target_img)
    file_name = [f for f in os.listdir(dir_img) if isfile(join(dir_img, f))][0]
    path_img = os.path.join(dir_img, file_name)
    path_target_img = os.path.join(dir_target_img, 'image.jpg')
    shutil.copyfile(path_img, path_target_img)

    path_intrinsics = os.path.join(dir_sun, sub_dir, 'intrinsics.txt')
    path_target_intrinsics = os.path.join(dir_target_img, 'intrinsics.txt')
    shutil.copyfile(path_intrinsics, path_target_intrinsics)

    dir_img = os.path.join(dir_sun, sub_dir, 'depth')
    file_name = [f for f in os.listdir(dir_img) if isfile(join(dir_img, f))][0]
    path_img = os.path.join(dir_img, file_name)
    path_target_img = os.path.join(dir_target_img, 'depth.png')
    shutil.copyfile(path_img, path_target_img)

    dir_img = os.path.join(dir_sun, sub_dir, 'depth_bfx')
    file_name = [f for f in os.listdir(dir_img) if isfile(join(dir_img, f))][0]
    path_img = os.path.join(dir_img, file_name)
    path_target_img = os.path.join(dir_target_img, 'depth_bfx.png')
    shutil.copyfile(path_img, path_target_img)

    dir_img = os.path.join(dir_sun_hha, sub_dir, 'hha')
    file_name = [f for f in os.listdir(dir_img) if isfile(join(dir_img, f))][0]
    path_img = os.path.join(dir_img, file_name)
    path_target_img = os.path.join(dir_target_img, 'hha.png')
    shutil.copyfile(path_img, path_target_img)

    dir_img = os.path.join(dir_sun_hha, sub_dir, 'hha_bfx')
    file_name = [f for f in os.listdir(dir_img) if isfile(join(dir_img, f))][0]
    path_img = os.path.join(dir_img, file_name)
    path_target_img = os.path.join(dir_target_img, 'hha_bfx.png')
    shutil.copyfile(path_img, path_target_img)

    dir_img = os.path.join(dir_sun_xyz, sub_dir, 'xyz')
    file_name = [f for f in os.listdir(dir_img) if isfile(join(dir_img, f))][0]
    path_img = os.path.join(dir_img, file_name)
    path_target_img = os.path.join(dir_target_img, 'xyz.bin')
    shutil.copyfile(path_img, path_target_img)

    dir_img = os.path.join(dir_sun_xyz, sub_dir, 'xyz_bfx')
    file_name = [f for f in os.listdir(dir_img) if isfile(join(dir_img, f))][0]
    path_img = os.path.join(dir_img, file_name)
    path_target_img = os.path.join(dir_target_img, 'xyz_bfx.bin')
    shutil.copyfile(path_img, path_target_img)


def make_sun_rgbd(dir_sun, dir_sun_hha, dir_sun_xyz, sub_dirs, seg_labels, dir_sun_target, test_num=0):
    print(len(sub_dirs), len(seg_labels))
    assert len(sub_dirs) == len(seg_labels)

    for i, sub_dir in enumerate(sub_dirs):
        print('id', i)
        if 0 < test_num <= i:
            break
        combine_imgdep_hha(dir_sun, dir_sun_hha, dir_sun_xyz, dir_sun_target, sub_dir)
        img_label = Image.fromarray(seg_labels[i])
        dir_label_out = os.path.join(dir_sun_target, 'data', sub_dir)
        if not os.path.exists(dir_label_out):
            os.makedirs(dir_label_out)
        img_label.save(os.path.join(dir_label_out, 'label.png'))

    return 0


if __name__ == '__main__':
    dir_sun_rgbd = "user_home/Disk/datasets/sun_rgbd/SUNRGBD"
    dir_sun_rgbd_hha = "user_home/Disk/datasets/sun_rgbd/sunrgbd_hha"
    dir_sun_rgbd_xyz = "user_home/Disk/datasets/sun_rgbd/sunrgbd_xyz"
    path_train_test_split = "user_home/Disk/datasets/sun_rgbd/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat"
    path_seg_label = "user_home/Disk/datasets/sun_rgbd/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat"
    path_seg_label_name = "user_home/Disk/datasets/sun_rgbd/SUNRGBDtoolbox/Metadata/seg37list.mat"
    dir_target = "user_home/Disk/datasets/sun_rgbd/sun_rgbd"

    test_num = 0
    # Load list of train,test then save
    sub_dirs_train, sub_dirs_test = load_list(path_train_test_split)
    np.savetxt(os.path.join(dir_target, "train_list.txt"), np.array(sub_dirs_train), fmt='%s')
    np.savetxt(os.path.join(dir_target, "test_list.txt"), np.array(sub_dirs_test), fmt='%s')
    sub_dirs = sub_dirs_test + sub_dirs_train

    # Load label name
    label_names = load_label_name(path_seg_label_name)
    label_names = ['unknown'] + label_names
    print(label_names)
    np.savetxt(os.path.join(dir_target, 'label_names.txt'), label_names, fmt='%s')

    # Load seg_label of train,test
    seg_labels = load_seg_label(path_seg_label)

    # For test
    if test_num > 0:
        print('sub_dirs_train:', len(sub_dirs_train), sub_dirs_train[0])
        print('sub_dirs_test:', len(sub_dirs_test), sub_dirs_test[0])
        print('label_names:', len(label_names), label_names)

        seg_label_min = 100
        seg_label_max = 0
        for seg_label in seg_labels:
            seg_label_min = min(seg_label_min, seg_label.min())
            seg_label_max = max(seg_label_max, seg_label.max())
        print('seg_labels min:', seg_label_min)
        print('seg_labels max:', seg_label_max)

    # The first 5050 images in the seg_labels contain labels for test dataset
    # while training set labels begin from 5051 and end at 10335.
    make_sun_rgbd(dir_sun_rgbd, dir_sun_rgbd_hha, dir_sun_rgbd_xyz, sub_dirs, seg_labels, dir_target, test_num)
