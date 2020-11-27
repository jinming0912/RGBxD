import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from data_preprocess.nyu_v2_fcn import data_utils


def make_h5(list_name_id, dir_fcn_data, path_h5):
    rgbs = []
    deps = []
    hhas = []
    labels = []
    for name_id in list_name_id:
        rgb = data_utils.load_image(dir_fcn_data, name_id)
        dep = data_utils.load_depth(dir_fcn_data, name_id)
        hha = data_utils.load_hha(dir_fcn_data, name_id)
        label = data_utils.load_label(dir_fcn_data, name_id)

        rgbs.append(rgb)
        deps.append(dep)
        hhas.append(hha)
        labels.append(label)

    h5file = h5py.File(path_h5, 'w')
    h5file.create_dataset('rgb', data=np.array(rgbs).astype(np.uint8))
    h5file.create_dataset('depth', data=np.array(deps).astype(np.float32))
    h5file.create_dataset('hha', data=np.array(hhas).astype(np.uint8))
    h5file.create_dataset('label', data=np.array(labels).astype(np.uint8))
    h5file.close()


def main():
    dir_fcn_data = "user_home/Disk/datasets/Downloads/NYU-Depth-V2/nyud"
    dir_tfrecord = "user_home/Disk/datasets/nyu_v2"
    list_train = data_utils.get_train_ids(dir_fcn_data)
    list_test = data_utils.get_test_ids(dir_fcn_data)
    print('list_train_length', len(list_train))
    print('list_test_length', len(list_test))

    path_h5 = os.path.join(dir_tfrecord, 'train_fcn.h5')
    make_h5(list_train, dir_fcn_data, path_h5)
    path_h5 = os.path.join(dir_tfrecord, 'test_fcn.h5')
    make_h5(list_test, dir_fcn_data, path_h5)


def check_h5(path_tfrecord):
    labels, rgbs, deps, hhas = data_utils.load_nyu_v2_fcn(path_tfrecord)
    dep = deps[0]
    plt.imshow(rgbs[0])
    plt.axis('off')
    plt.show()
    plt.imshow(deps[0])
    plt.axis('off')
    plt.show()
    plt.imshow(hhas[0])
    plt.axis('off')
    plt.show()
    plt.imshow(labels[0])
    plt.axis('off')
    plt.show()


def test_img_aug():
    import imgaug as ia
    from imgaug import augmenters as iaa

    labels, rgbs, deps, hhas = data_utils.load_nyu_v2_fcn("user_home/Disk/datasets/nyu_v2/train_fcn.h5")
    print('labels.shape', labels.shape)  # (795, 425, 560)
    rgbs = rgbs[0:2]
    labels = labels[0:2]
    deps = deps[0:2]
    hhas = hhas[0:2]
    plt.imshow(rgbs[0])
    # plt.axis('off')
    plt.show()
    plt.imshow(deps[0])
    # plt.axis('off')
    plt.show()
    plt.imshow(labels[0])
    # plt.axis('off')
    plt.show()

    labels_ia = []
    for label in labels:
        labels_ia.append(ia.SegmentationMapOnImage(label, shape=label.shape, nb_classes=41))

    seq = iaa.SomeOf((0, 3), [
        iaa.Crop(px=(0, 100)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(1.0),  # 0.5 is the probability, horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 3.0)),  # blur images with a sigma of 0 to 3.0
        iaa.Dropout([0.05, 0.2]),  # drop 5% or 20% of all pixels
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects heatmaps)
        iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects heatmaps)
    ], random_order=True)

    for i in range(0, 10):
        seq_det = seq.to_deterministic()
        rgbs_aug = seq_det.augment_images(rgbs)
        plt.imshow(rgbs_aug[0])
        # plt.axis('off')
        plt.show()

        deps_aug = seq_det.augment_images(deps)
        plt.imshow(deps_aug[0])
        # plt.axis('off')
        plt.show()

        labels_aug = seq_det.augment_segmentation_maps(labels_ia)
        plt.imshow(labels_aug[0].get_arr_int().astype(np.uint8))
        # plt.axis('off')
        plt.show()


if __name__ == '__main__':

    check_h5("user_home/Disk/datasets/nyu_v2/test_fcn.h5")
