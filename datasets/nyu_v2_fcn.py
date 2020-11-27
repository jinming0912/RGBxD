import os
from PIL import Image
import numpy as np
import imgaug as ia

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from datasets.base_dataset import BaseDataset
from utils import preprocess_utils


class NyuV2Fcn(BaseDataset):
    def __init__(self,
                 batch_size,
                 img_size=[560, 425],
                 min_resize_value=None,
                 max_resize_value=None,
                 resize_factor=None,
                 min_scale_factor=1.,
                 max_scale_factor=1.,
                 scale_factor_step_size=0,
                 ignore_label=0,
                 base_dir=os.path.join(os.environ['HOME'], "Datasets/Downloads/NYU-Depth-V2/nyud"),
                 with_aug_flip=True,
                 with_aug_scale_crop=True,
                 with_aug_gaus_noise=True):
        super(NyuV2Fcn, self).__init__(name='nyu_v2_fcn')
        # self.mean_bgr = (122.675, 116.669, 104.008)
        self.mean_rgb = [103.06, 115.90, 123.15]
        self.mean_hha = [132.431, 94.076, 118.477]
        self.mean_dep = [0]
        self.mean_xyz = [0.0, 0.0, 0.0]
        self.img_size = img_size  # [width, height]
        self.num_class = 41
        # self.calss_weights = [0.0] + [1.0] * (self.num_class - 1)

        self.batch_size = batch_size
        self.crop_height = self.img_size[1]
        self.crop_width = self.img_size[0]
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.resize_factor = resize_factor
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_factor_step_size = scale_factor_step_size
        self.ignore_label = ignore_label
        self.base_dir = base_dir
        self.with_aug_flip = with_aug_flip
        self.with_aug_scale_crop = with_aug_scale_crop
        self.with_aug_gaus_noise = with_aug_gaus_noise

        self.sample_weight = np.array([0.0] + [1.0] * (self.num_class - 1))[:, np.newaxis]

    def create_train_dataset(self, repeat=True):
        dataset = tf.data.Dataset.from_tensor_slices(self.get_trainval_list())
        dataset = dataset.shuffle(self.get_trainval_size())
        dataset = dataset.map(self.get_single_data_tf)
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.prefetch(buffer_size=10)
        return dataset

    def create_test_dataset(self, one_batch=False):
        dataset = tf.data.Dataset.from_tensor_slices(self.get_test_list())
        dataset = dataset.map(self.get_test_data_tf)
        if one_batch:
            dataset = dataset.batch(1)
        else:
            dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.prefetch(buffer_size=10)
        return dataset

    # Map fun tf
    def get_single_data_tf(self, img_id):
        # print('img_id', img_id)
        # tf.print(img_id)
        img_id = tf.as_string(img_id)
        img = self.read_single_img_tf(img_id)
        hha = self.read_single_hha_tf(img_id)
        depth = self.read_single_depth_tf(img_id)
        xyz = self.read_single_xyz_tf(img_id)
        label = self.read_single_label_tf(img_id)

        image = tf.concat([img, hha, depth, xyz], axis=-1)
        processed_image, label = self.aug_single_data_tf(image, label)
        return processed_image, label

    def get_test_data_tf(self, img_id):
        img_id = tf.as_string(img_id)
        rgb = self.read_single_img_tf(img_id)
        hha = self.read_single_hha_tf(img_id)
        dep = self.read_single_depth_tf(img_id)
        xyz = self.read_single_xyz_tf(img_id)
        label = self.read_single_label_tf(img_id)

        processed_image = tf.concat([rgb, hha, dep, xyz], axis=-1)
        if self.img_size[0] != 560 or self.img_size[1] != 425:
            processed_image = tf.compat.v1.image.resize(processed_image, (self.crop_height, self.crop_width),
                                                        method=ResizeMethod.BILINEAR,
                                                        align_corners=True)
            label = tf.compat.v1.image.resize(label, (self.crop_height, self.crop_width),
                                              method=ResizeMethod.NEAREST_NEIGHBOR,
                                              align_corners=True)
        processed_image = self.normal_data(processed_image)
        return processed_image, label

    def aug_single_data_tf(self, image, label):
        processed_image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.int32)

        if self.with_aug_scale_crop:
            # Resize image and label to the desired range.
            if self.min_resize_value or self.max_resize_value:
                [processed_image, label] = (
                    preprocess_utils.resize_to_range(
                        image=processed_image,
                        label=label,
                        min_size=self.min_resize_value,
                        max_size=self.max_resize_value,
                        factor=self.resize_factor))

            # Data augmentation by randomly scaling the inputs.
            scale = preprocess_utils.get_random_scale(self.min_scale_factor,
                                                      self.max_scale_factor,
                                                      self.scale_factor_step_size)
            processed_image, label = preprocess_utils.randomly_scale_image_and_label(
                processed_image, label, scale)
            processed_image.set_shape([None, None, 10])

            # Pad image and label to have dimensions >= [crop_height, crop_width]
            image_shape = tf.shape(processed_image)
            image_height = image_shape[0]
            image_width = image_shape[1]

            target_height = image_height + tf.maximum(self.crop_height - image_height, 0)
            target_width = image_width + tf.maximum(self.crop_width - image_width, 0)
            # offset_height = (target_height - image_height) // 2
            # offset_width = (target_width - image_width) // 2
            offset_height = 0
            offset_width = 0

            # Pad image with mean pixel value.
            mean_pixel = tf.reshape(self.mean_rgb + self.mean_hha + self.mean_dep + self.mean_xyz, [1, 1, 10])
            processed_image = preprocess_utils.pad_to_bounding_box(
                processed_image, offset_height, offset_width, target_height, target_width, mean_pixel)

            label = preprocess_utils.pad_to_bounding_box(
                label, offset_height, offset_width, target_height, target_width, self.ignore_label)

            # Randomly crop the image and label.
            processed_image, label = preprocess_utils.random_crop(
                [processed_image, label], self.crop_height, self.crop_width)

            processed_image.set_shape([self.crop_height, self.crop_width, 10])
            label.set_shape([self.crop_height, self.crop_width, 1])

        # Randomly left-right flip the image and label.
        if self.with_aug_flip:
            processed_image, label, _ = preprocess_utils.random_flip([processed_image, label], prob=0.5)

        # Noise
        if self.with_aug_gaus_noise:
            def tf_rand(minval=0., maxval=1., dtype=tf.float32):
                return tf.reduce_sum(tf.random.uniform(shape=[1], minval=minval, maxval=maxval, dtype=dtype))

            rgb, hha, dep, xyz = tf.split(axis=-1, num_or_size_splits=[3, 3, 1, 3], value=processed_image)
            hsv = tf.image.rgb_to_hsv(rgb)
            hue, sat, val = tf.split(axis=-1, num_or_size_splits=3, value=hsv)

            hue = tf.keras.backend.clip(hue + tf_rand() * 70 - 35, 0, 360.)
            sat = tf.keras.backend.clip(sat + tf_rand() * 0.3 - 0.15, 0, 1.)
            val = tf.keras.backend.clip(val + tf_rand() * 50 - 25, 0, 255.)

            hsv = tf.concat([hue, sat, val], axis=-1)
            rgb = tf.image.hsv_to_rgb(hsv)

            processed_image = tf.concat([rgb, hha, dep, xyz], axis=-1)

        processed_image = self.normal_data(processed_image)
        return processed_image, label

    def normal_data(self, processed_image):
        red, green, blue, hd, hg, angle, dep, xyz = tf.split(num_or_size_splits=[1, 1, 1, 1, 1, 1, 1, 3],
                                                             value=processed_image, axis=-1)
        # Convert RGB to BGR
        bgr = tf.concat(axis=-1, values=[
            blue - self.mean_rgb[2],
            green - self.mean_rgb[1],
            red - self.mean_rgb[0],
        ])

        # HHA preprocess
        hha = tf.concat(axis=-1, values=[
            hd - self.mean_hha[0],
            hg - self.mean_hha[1],
            angle - self.mean_hha[2],
        ])

        dep = dep / 120.

        inputs = tf.concat([bgr, hha, dep, xyz], axis=-1)
        return inputs

    # Get list
    def get_trainval_list(self):
        return self.get_list('trainval')

    def get_test_list(self):
        return self.get_list('test')

    def get_list(self, type):
        path_trainval_list = os.path.join(self.base_dir, type + '.txt')
        return np.loadtxt(path_trainval_list, dtype=np.int)

    # Get Path tf
    def get_img_real_path_tf(self, img_id):
        return self.get_real_path_tf('images', img_id)

    def get_hha_real_path_tf(self, img_id):
        return self.get_real_path_tf('hha', img_id)

    def get_depth_real_path_tf(self, img_id):
        return self.get_real_path_tf('depth', img_id)

    def get_xyz_real_path_tf(self, img_id):
        return self.get_real_path_tf('xyz', img_id)

    def get_label_real_path_tf(self, img_id):
        return self.get_real_path_tf('label', img_id)

    def get_real_path_tf(self, type, img_id):
        data_path = tf.convert_to_tensor(os.path.join(self.base_dir, 'data', type, 'img_'), dtype=tf.string)
        data_extensions = tf.convert_to_tensor('.txt' if type == 'xyz' else '.png', dtype=tf.string)
        return tf.strings.join([data_path, img_id, data_extensions])

    # Read png tf
    def read_single_img_tf(self, img_id):
        img = self.read_single_png_tf(self.get_img_real_path_tf(img_id), 3)
        img = tf.cast(img, tf.float32)
        return img

    def read_single_hha_tf(self, img_id):
        img = self.read_single_png_tf(self.get_hha_real_path_tf(img_id), 3)
        img = tf.cast(img, tf.float32)
        return img

    def read_single_depth_tf(self, img_id):
        img = self.read_single_png_tf(self.get_depth_real_path_tf(img_id), 1, dtype=tf.uint16)
        img = tf.cast(img, tf.float32)
        return img

    def read_single_label_tf(self, img_id):
        label = self.read_single_png_tf(self.get_label_real_path_tf(img_id), 1)
        label = tf.cast(label, tf.int32)
        return label

    def read_single_png_tf(self, path, channels, dtype=tf.uint8):
        return tf.image.decode_png(tf.io.read_file(path), dtype=dtype, channels=channels)

    def read_single_xyz_tf(self, img_id):
        xyz = self.read_single_txt_tf(self.get_xyz_real_path_tf(img_id))
        xyz = tf.reshape(xyz, [425, 560, 3])
        xyz = tf.cast(xyz, tf.float32)
        return xyz

    def read_single_txt_tf(self, path):
        tsr_f = tf.io.read_file(path)
        tsr_s = tf.strings.regex_replace(tsr_f, pattern='\n', rewrite=' ')
        tsr_s = tf.strings.strip(tsr_s)
        tsr_s = tf.strings.split(tsr_s, sep=' ')
        tsr_i = tf.strings.to_number(tsr_s, out_type=tf.float32)
        return tsr_i

    # Data num
    def get_trainval_size(self):
        return self.get_trainval_list().size

    def get_test_size(self):
        return self.get_test_list().size

    def get_input_shape(self):
        return [self.crop_height, self.crop_width, 10]
