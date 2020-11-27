import numpy as np
import os
import matplotlib.pyplot as plt
from data_preprocess.nyu_v2_fcn import data_utils
import tensorflow as tf


def make_tfrecord(list_name_id, dir_fcn_data, path_tfrecord):
    options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(path_tfrecord, options=options_zlib)
    for name_id in list_name_id:
        rgb = data_utils.load_image(dir_fcn_data, name_id)
        dep = data_utils.load_depth(dir_fcn_data, name_id)
        hha = data_utils.load_hha(dir_fcn_data, name_id)
        label = data_utils.load_label(dir_fcn_data, name_id)

        example = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[rgb.shape[0], rgb.shape[1]])),
            'rgb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb.tostring()])),
            'dep': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dep.tostring()])),
            'hha': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hha.tostring()])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
        }))
        writer.write(example.SerializeToString())

    writer.close()


def main():
    dir_fcn_data = "user_home/Disk/datasets/Downloads/NYU-Depth-V2/nyud"
    dir_tfrecord = "user_home/Disk/datasets/nyu_v2"
    list_train = np.loadtxt(os.path.join(dir_fcn_data, 'trainval.txt'), dtype=np.str)
    list_test = np.loadtxt(os.path.join(dir_fcn_data, 'test.txt'), dtype=np.str)
    print('list_train_length', len(list_train))
    print('list_test_length', len(list_test))

    path_tfrecord = os.path.join(dir_tfrecord, 'train.tfrecord')
    make_tfrecord(list_train, dir_fcn_data, path_tfrecord)
    path_tfrecord = os.path.join(dir_tfrecord, 'test.tfrecord')
    make_tfrecord(list_test, dir_fcn_data, path_tfrecord)


def check_tfrecord(path_tfrecord):
    rgb, dep, hha, label = data_utils.read_tfrecord(path_tfrecord)
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()
    plt.imshow(dep)
    plt.axis('off')
    plt.show()
    plt.imshow(hha)
    plt.axis('off')
    plt.show()
    plt.imshow(label)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # main()
    check_tfrecord("user_home/Disk/datasets/nyu_v2/train.tfrecord")
    check_tfrecord("user_home/Disk/datasets/nyu_v2/test.tfrecord")