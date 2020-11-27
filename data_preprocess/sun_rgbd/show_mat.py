import scipy.io as scio
from matplotlib import pyplot as plt
import h5py
import numpy as np
import copy
# dataFile = "user_home/Disk/datasets/sun_rgbd/SUNRGBDtoolbox/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat"
# dataFile = "user_home/Disk/datasets/sun_rgbd/SUNRGBD/kv1/NYUdata/NYU0546/seg.mat"
# data = scio.loadmat(dataFile)
# print(data.keys())
# print(data['alltrain'].shape)
# print(data['alltest'].shape)
# print(data['trainvalsplit'].shape)
# print(data['names'])
# print(data['seglabel'].shape)
#
# img = data['seglabel'].reshape(data['seglabel'].shape[0], data['seglabel'].shape[1], 1)
# img_2d = data['seglabel']
# print(img.shape)
# plt.imshow(data['seglabel'])
# plt.show()


def show_seg_h5():
    seg_file = "user_home/Disk/datasets/sun_rgbd/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat"
    with h5py.File(seg_file, 'r') as f:
        print(list(f.keys()))

        seg37list = f['seg37list']
        print(np.array(f[seg37list[0][0]]).tostring().decode('utf-8'))

        seglistall = f['seglistall']
        print(seglistall)

        SUNRGBD2Dseg = f['SUNRGBD2Dseg']
        print(list(SUNRGBD2Dseg.keys()))
        seglabel = SUNRGBD2Dseg['seglabel']
        mat_label = np.array(SUNRGBD2Dseg[seglabel[0][0]])
        print(mat_label.shape)
        mat_label = mat_label.swapaxes(0, 1)
        print(mat_label.shape)
        plt.imshow(mat_label)
        plt.show()


def show_seg_list():
    seg_list_file = "user_home/Disk/datasets/sun_rgbd/SUNRGBDtoolbox/Metadata/seg37list.mat"
    data = scio.loadmat(seg_list_file)
    print(data.keys())
    seg37list = data['seg37list']
    print(seg37list.shape)


def show_allaplit():
    split_file = "user_home/Disk/datasets/sun_rgbd/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat"
    data = scio.loadmat(split_file)
    print(data.keys())
    alltrain = data['alltrain']
    alltest = data['alltest']
    trainvalsplit = data['trainvalsplit']
    print('alltrain', alltrain.shape, alltrain[0][1][0])
    print('alltest', alltest.shape, alltest)
    # with h5py.File(split_file, 'r') as f:
    #     print(list(f.keys()))


if __name__ == '__main__':

    show_allaplit()
