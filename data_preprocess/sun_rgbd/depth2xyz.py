import os
import struct

import numpy as np
import imageio
import math
from os import listdir
from os.path import isfile, join
import sys
sys.path.append("user_home/Code/python/project/Segmentatron")

from data_preprocess.sun_rgbd.camera import processCamMat, getCameraParam
from data_preprocess.sun_rgbd.rgbd_util import *
from data_preprocess.sun_rgbd.utils import load_list
from multiprocessing import Process


def depth2xyz(dir_sun, dir_sun_xyz_out, sub_dirs):
    for idx, sub_dir in enumerate(sub_dirs):
        print('idx', idx, sub_dir)
        path_cam_mat = join(dir_sun, sub_dir, 'intrinsics.txt')
        with open(path_cam_mat, 'r') as camf:
            camera_matrix = processCamMat(camf.readlines())
            # camera_matrix = getCameraParam('color')

        dir_depth = join(dir_sun, sub_dir, 'depth_bfx')
        dir_raw_depth = join(dir_sun, sub_dir, 'depth')

        path_depth = [join(dir_depth, f) for f in listdir(dir_depth) if isfile(join(dir_depth, f))][0]
        path_raw_depth = [join(dir_raw_depth, f) for f in listdir(dir_raw_depth) if isfile(join(dir_raw_depth, f))][0]

        depth = imageio.imread(path_depth).astype(float) / 10000
        raw_depth = imageio.imread(path_raw_depth).astype(float) / 10000  # meter
        print(depth.shape)

        x, y, z = getPointCloudFromZ(raw_depth * 100, camera_matrix, 1)
        print(x.shape, y.shape, z.shape)
        xyz = np.concatenate([x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]], axis=-1)
        x_bfx, y_bfx, z_bfx = getPointCloudFromZ(depth * 100, camera_matrix, 1)
        xyz_bfx = np.concatenate([x_bfx[:, :, np.newaxis], y_bfx[:, :, np.newaxis], z_bfx[:, :, np.newaxis]], axis=-1)

        print(xyz.shape, xyz[0], xyz[-1])

        dir_xyz_out = join(dir_sun_xyz_out, sub_dir, 'xyz')
        if not os.path.exists(dir_xyz_out):
            os.makedirs(dir_xyz_out)

        print(np.array(xyz.shape, dtype=np.float32))
        print(np.array(xyz.shape, dtype=np.float32).flatten())
        return
        xyz_data = np.concatenate([np.array(xyz.shape, dtype=np.float32).flatten(), xyz.flatten()])
        xyz_buf = struct.pack('=%sf' % xyz_data.size, *xyz_data)
        with open(join(dir_xyz_out, 'xyz.bin'), 'wb') as f:
            f.write(xyz_buf)

        dir_xyz_bfx_out = join(dir_sun_xyz_out, sub_dir, 'xyz_bfx')
        if not os.path.exists(dir_xyz_bfx_out):
            os.makedirs(dir_xyz_bfx_out)
        xyz_bfx_data = np.concatenate([np.array(xyz_bfx.shape, dtype=np.float32).flatten(), xyz_bfx.flatten()])
        xyz_bfx_buf = struct.pack('=%sf' % xyz_bfx_data.size, *xyz_bfx_data)
        with open(join(dir_xyz_bfx_out, 'xyz_bfx.bin'), 'wb') as f:
            f.write(xyz_bfx_buf)

        # return


def main():
    dir_sun = 'user_home/Disk/datasets/sun_rgbd/SUNRGBD'
    dir_sun_xyz_out = 'user_home/Disk/datasets/sun_rgbd/sunrgbd_xyz'
    path_train_test_split = "user_home/Disk/datasets/sun_rgbd/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat"

    # Load list of train,test then save
    sub_dirs_train, sub_dirs_test = load_list(path_train_test_split)
    sub_dirs = sub_dirs_test + sub_dirs_train


    def chunkIt(seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    num_process = 1
    multi_sub_dirs = chunkIt(sub_dirs, num_process)
    processes = []
    for dirs in multi_sub_dirs:
        p = Process(target=depth2xyz, args=(dir_sun, dir_sun_xyz_out, dirs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()