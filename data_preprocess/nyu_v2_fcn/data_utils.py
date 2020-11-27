import os

import h5py

import numpy as np
from PIL import Image
import scipy.io
import tensorflow as tf


def load_image(nyud_dir, idx):
    """
    Load input image and preprocess for Caffe:
    - transpose to channel x height x width order
    """
    im = Image.open('{}/data/images/img_{}.png'.format(nyud_dir, idx))
    in_ = np.array(im)
    return in_


def load_label(nyud_dir, idx):
    """
    Load label image as 1 x height x width integer array of label indices.
    Shift labels so that classes are 0-39 and void is 255 (to ignore it).
    The leading singleton dimension is required by the loss.
    """
    label = scipy.io.loadmat('{}/segmentation/img_{}.mat'.format(nyud_dir, idx))['segmentation'].astype(np.uint8)
    return label


def load_depth(nyud_dir, idx):
    """
    Load pre-processed depth for NYUDv2 segmentation set.
    """
    im = Image.open('{}/data/depth/img_{}.png'.format(nyud_dir, idx))
    d = np.array(im)
    return d


def load_hha(nyud_dir, idx):
    im = Image.open('{}/data/hha/img_{}.png'.format(nyud_dir, idx))
    hha = np.array(im)
    return hha


def get_train_ids(nyud_dir):
    return np.loadtxt(os.path.join(nyud_dir, 'trainval.txt'), dtype=np.str)


def get_test_ids(nyud_dir):
    return np.loadtxt(os.path.join(nyud_dir, 'test.txt'), dtype=np.str)


def get_all_ids(nyud_dir):
    train_ids = get_train_ids(nyud_dir)
    test_ids = get_test_ids(nyud_dir)
    return np.concatenate((train_ids, test_ids))


def write_tfrecord(rgb, dep, hha, label, path_tfrecord_out):
    rgb_data = rgb.tostring()
    print(len(rgb_data))
    options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    writer = tf.python_io.TFRecordWriter(path_tfrecord_out, options=options_zlib)
    example = tf.train.Example(features=tf.train.Features(feature={
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[rgb.shape[0], rgb.shape[1]])),
        'rgb': tf.train.Feature(bytes_list=tf.train.BytesList(value=[rgb.tostring()])),
        'dep': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dep.tostring()])),
        'hha': tf.train.Feature(bytes_list=tf.train.BytesList(value=[hha.tostring()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()]))
    }))
    writer.write(example.SerializeToString())
    writer.close()


def read_tfrecord(path_tfrecord):
    def parse_record(example_proto):
        features = {
            'shape': tf.FixedLenFeature([2], tf.int64),
            'rgb': tf.FixedLenFeature([], tf.string),
            'dep': tf.FixedLenFeature([], tf.string),
            'hha': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.parse_single_example(example_proto, features=features)
        shape = parsed_features['shape']
        rgb = tf.decode_raw(parsed_features['rgb'], tf.uint8)
        # print('shape', shape)
        rgb = tf.reshape(rgb, [425, 560, 3])
        # rgb = tf.cast(rgb, tf.float32)
        label = tf.decode_raw(parsed_features['label'], tf.uint8)
        label = tf.reshape(label, [425, 560])
        label = tf.cast(label, tf.int32)
        return rgb, label

    dataset = tf.data.TFRecordDataset(path_tfrecord, compression_type='ZLIB')
    dataset = dataset.map(parse_record)
    iterator = dataset.make_one_shot_iterator()

    with tf.Session() as sess:
        rgb, label = sess.run(iterator.get_next())

    return rgb, rgb, rgb, label


def load_nyu_v2_fcn(path_data):
    data = h5py.File(path_data, 'r')
    rgbs = data['rgb'].value
    deps = data['depth'].value
    hhas = data['hha'].value
    labels = data['label'].value
    return labels, rgbs, deps, hhas
