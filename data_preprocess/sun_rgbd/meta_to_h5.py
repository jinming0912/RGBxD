import h5py
import os
import numpy as np
import scipy.io as scio
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def load_list(path_split):
    data = scio.loadmat(path_split)
    sub_dirs_train = [dir_scene[0].replace("/n/fs/sun3d/data/", "") for dir_scene in data['alltrain'][0]]
    sub_dirs_test = [dir_scene[0].replace("/n/fs/sun3d/data/", "") for dir_scene in data['alltest'][0]]
    return sub_dirs_train, sub_dirs_test


def load_label_name(path_label_name):
    data = scio.loadmat(path_label_name)
    return [name[0] for name in data['seg37list'][0]]


def load_seg_label(path_seg_label):
    with h5py.File(path_seg_label, 'r') as f:
        SUNRGBD2Dseg = f['SUNRGBD2Dseg']
        seglabel = SUNRGBD2Dseg['seglabel']
        seg_labels = [np.array(SUNRGBD2Dseg[idx[0]]).swapaxes(0, 1).astype(np.uint8)[:, :, np.newaxis]
                      for idx in seglabel]
    return seg_labels


def load_one_img(dir_img):
    path_img = None
    for file_name in os.listdir(dir_img):
        if file_name.count('.png') > 0 or file_name.count('.jpg') > 0:
            path_img = os.path.join(dir_img, file_name)

    if path_img is not None:
        img = mpimg.imread(path_img)
    else:
        print('load_one_img failed!')
        img = None
    return img


def meta_to_h5(root_dir, sub_dirs, seg_labels, dir_npy, test_num=0):
    assert len(sub_dirs) == len(seg_labels)

    rgbs = []
    deps = []
    for i, sub_dir in enumerate(sub_dirs):
        if 0 < test_num <= i:
            break

        rgb = load_one_img(os.path.join(root_dir, sub_dir, 'image'))
        dep = load_one_img(os.path.join(root_dir, sub_dir, 'depth_bfx'))
        if rgb is not None and dep is not None:
            rgbs.append(rgb)
            dep = dep[:, :, np.newaxis]
            deps.append(dep)
        else:
            return -1

        if test_num > 0:

            plt.imshow(rgb)
            plt.axis('off')
            plt.show()
            plt.imshow(dep)
            plt.axis('off')
            plt.show()
            plt.imshow(seg_labels[i])
            plt.axis('off')
            plt.show()

    path_rgbs = os.path.join(dir_npy, 'rgbs.npy')
    path_deps = os.path.join(dir_npy, 'deps.npy')
    path_labels = os.path.join(dir_npy, 'labels.npy')
    np.save(path_rgbs, np.array(rgbs))
    np.save(path_deps, np.array(deps))
    np.save(path_labels, np.array(seg_labels))

    return 0


def check_npy(path_h5):
    rgbs = np.load(os.path.join(path_h5, 'rgbs.npy'))
    deps = np.load(os.path.join(path_h5, 'deps.npy'))
    labels = np.load(os.path.join(path_h5, 'labels.npy'))

    rgb_max = max([rgb_mat.flatten().max() for rgb_mat in rgbs])
    rgb_min = min([rgb_mat.flatten().min() for rgb_mat in rgbs])
    print("rgb shape", rgbs[-1].shape)
    print('rgbs dtype, max, min:', rgbs.shape, rgbs.dtype, rgb_max, rgb_min)
    dep_max = max([dep_mat.flatten().max() for dep_mat in deps])
    dep_min = min([dep_mat.flatten().min() for dep_mat in deps])
    print("dep shape", deps[-1].shape)
    print('deps dtype, max, min:', deps.shape, deps.dtype, dep_max, dep_min)
    label_max = max([label_mat.flatten().max() for label_mat in labels])
    label_min = min([label_mat.flatten().min() for label_mat in labels])
    print("label shape", labels[-1].shape)
    print('labels dtype, max, min:', labels.shape, labels.dtype, label_max, label_min)

    # plt.imshow(rgbs[-1])
    # plt.axis('off')
    # plt.show()
    # plt.imshow(deps[-1])
    # plt.axis('off')
    # plt.show()
    # plt.imshow(labels[-1])
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    dir_sun_rgbd = "user_home/Disk/datasets/sun_rgbd"
    path_train_test_split = os.path.join(dir_sun_rgbd, "SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat")
    path_seg_label = os.path.join(dir_sun_rgbd, "SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat")
    path_seg_label_name = os.path.join(dir_sun_rgbd, "SUNRGBDtoolbox/Metadata/seg37list.mat")

    dir_npy_train = os.path.join(dir_sun_rgbd, "train")
    dir_npy_test = os.path.join(dir_sun_rgbd, "test")

    test_num = 1

    # Load list of train,test
    sub_dirs_train, sub_dirs_test = load_list(path_train_test_split)

    # Load label name
    label_names = load_label_name(path_seg_label_name)
    label_names = ['unknown'] + label_names
    print(label_names)
    np.savetxt(os.path.join(dir_sun_rgbd, 'label_names.txt'), label_names, fmt='%s')

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
    meta_to_h5(dir_sun_rgbd, sub_dirs_train, seg_labels[5050:], dir_npy_train, test_num)
    meta_to_h5(dir_sun_rgbd, sub_dirs_test, seg_labels[:5050], dir_npy_test, test_num)

    check_npy(dir_npy_train)
    check_npy(dir_npy_test)