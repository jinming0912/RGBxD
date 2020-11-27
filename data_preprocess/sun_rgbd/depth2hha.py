import os

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

'''
C: Camera matrix
D: Depth image, the unit of each element in it is "meter"
RD: Raw depth image, the unit of each element in it is "meter"
'''


def getHHA(C, D, RD):
    missingMask = (RD == 0)
    pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C)

    tmp = np.multiply(N, yDir)
    acosValue = np.minimum(1, np.maximum(-1, np.sum(tmp, axis=2)))
    angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
    angle = np.reshape(angle, h.shape)

    '''
    Must convert nan to 180 as the MATLAB program actually does. 
    Or we will get a HHA image whose border region is different
    with that of MATLAB program's output.
    '''
    angle[np.isnan(angle)] = 180

    pc[:, :, 2] = np.maximum(pc[:, :, 2], 100)
    I = np.zeros(pc.shape)

    # opencv-python save the picture in BGR order.
    I[:, :, 0] = 31000 / pc[:, :, 2]
    I[:, :, 1] = h
    I[:, :, 2] = (angle + 128 - 90)

    # print(np.isnan(angle))

    '''
    np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
    So I convert it to integer myself.
    '''
    I = np.rint(I)

    # np.uint8: 256->1, but in MATLAB, uint8: 256->255
    I[I > 255] = 255
    HHA = I.astype(np.uint8)
    return HHA


def depth2hha(dir_sun, dir_sun_hha_out, sub_dirs):
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
        raw_depth = imageio.imread(path_raw_depth).astype(float) / 10000

        hha = getHHA(camera_matrix, depth, raw_depth)
        hha_bfx = getHHA(camera_matrix, depth, depth)

        dir_hha_out = join(dir_sun_hha_out, sub_dir, 'hha')
        if not os.path.exists(dir_hha_out):
            os.makedirs(dir_hha_out)
        imageio.imwrite(join(dir_hha_out, 'hha.png'), hha)

        dir_hha_bfx_out = join(dir_sun_hha_out, sub_dir, 'hha_bfx')
        if not os.path.exists(dir_hha_bfx_out):
            os.makedirs(dir_hha_bfx_out)
        imageio.imwrite(join(dir_hha_bfx_out, 'hha_bfx.png'), hha_bfx)

def main():
    dir_sun = 'user_home/Disk/datasets/sun_rgbd/SUNRGBD'
    dir_sun_hha_out = 'user_home/Disk/datasets/sun_rgbd/sunrgbd_hha'
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

    num_process = 16
    multi_sub_dirs = chunkIt(sub_dirs, num_process)
    processes = []
    for dirs in multi_sub_dirs:
        p = Process(target=depth2hha, args=(dir_sun, dir_sun_hha_out, dirs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
