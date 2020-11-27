# -*- coding: utf-8 -*-

""" Deeplabv3+ model for Keras.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import numpy as np
import h5py
import tensorflow as tf

from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Concatenate
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import DepthwiseConv2D
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from tensorflow.python.keras.utils.data_utils import get_file
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.activations import relu
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input

import layers as ilayers

WEIGHTS_PATH_X = "url/keras-deeplab-v3-plus/releases/download/1.1" \
                 "/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5 "
WEIGHTS_PATH_MOBILE = "url/keras-deeplab-v3-plus/releases/download/1.1" \
                      "/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5 "
WEIGHTS_PATH_X_CS = "url/keras-deeplab-v3-plus/releases/download/1.2" \
                    "/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5 "
WEIGHTS_PATH_MOBILE_CS = "url/keras-deeplab-v3-plus/releases/download/1.2" \
                         "/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5 "


def DeeplabV3Plus(input_shape, batch_size, backbone, fusion_mode, num_class, weights,
                  with_pre_train_entry_flow_conv1_1=False,
                  with_pre_train_logits_semantic=False,
                  batch_normal_trainable=True,
                  conv_trainable=True,
                  experiment=None,
                  name='DeeplabV3Plus'):
    """ Instantiates the Deeplabv3+ architecture

    Optionally loads weights pre-trained
    on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
    # Arguments
        weights: one of 'pascal_voc' (pre-trained on pascal voc),
            'cityscapes' (pre-trained on cityscape) or None (random initialization)
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: shape of input image. format HxWxC
            PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
        classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
            If number of classes not aligned with the weights used, last layer is initialized randomly
        backbone: backbone to use. one of {'xception','mobilenetv2'}
        activation: optional activation to add to the top of the network.
            One of 'softmax', 'sigmoid' or None
        OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
            Used only for xception backbone.
        alpha: controls the width of the MobileNetV2 network. This is known as the
            width multiplier in the MobileNetV2 paper.
                - If `alpha` < 1.0, proportionally decreases the number
                    of filters in each layer.
                - If `alpha` > 1.0, proportionally increases the number
                    of filters in each layer.
                - If `alpha` = 1, default number of filters from the paper
                    are used at each layer.
            Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.

    # Returns
        A Keras model instance.

    # Raises
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
        ValueError: in case of invalid argument for `weights` or `backbone`

    """
    # if not (weights in {'pascal_voc', 'cityscapes', 'imagenet', None}):
    #     raise ValueError('The `weights` argument should be either '
    #                      '`None` (random initialization), `pascal_voc`, `cityscapes` or `imagenet` '
    #                      '(pre-trained on PASCAL VOC)')

    if not (backbone in {'xception', 'mobilenetv2', 'vgg16'}):
        raise ValueError('The `backbone` argument should be either '
                         '`xception`, `mobilenetv2` or `vgg16` ')
    if weights == 'pascal_voc':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_X,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_MOBILE,
                                    cache_subdir='models')
        file_weights = h5py.File(weights_path, 'r')
    elif weights == 'cityscapes':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_X_CS,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_MOBILE_CS,
                                    cache_subdir='models')
        file_weights = h5py.File(weights_path, 'r')
    else:
        if os.path.exists(weights):
            weights_path = weights
        else:
            weights_path = None
        file_weights = None

    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)

    # bgr, hha, dep = tf.split(axis=-1, num_or_size_splits=[3, 3, 1], value=inputs)
    # xyz = None
    bgr, hha, dep, xyz = tf.split(axis=-1, num_or_size_splits=[3, 3, 1, 3], value=inputs)
    if fusion_mode == 'bgr':
        x = bgr
    elif fusion_mode == 'hha':
        x = hha
    elif fusion_mode == 'bgr_hha':
        x = tf.concat([bgr, hha], axis=-1)
    elif fusion_mode in {'bgr_hha_xyz', 'bgr_hha_gw'}:
        x = tf.concat([bgr, hha, xyz], axis=-1)
    else:
        logging.error("Unknown model mode.")
        return None

    if backbone == 'xception':
        OS = 16
        atrous_rates, x, skip1, gw_weight = Inception(x, fusion_mode, weights, file_weights,
                                                      with_pre_train_entry_flow_conv1_1,
                                                      batch_normal_trainable=batch_normal_trainable,
                                                      conv_trainable=conv_trainable)
    elif backbone == 'mobilenetv2':
        OS = 8
        x = MobilenetV2(x, alpha=1., batch_normal_trainable=batch_normal_trainable, conv_trainable=conv_trainable)
    else:
        logging.error('Unknown backbone.')
        return None

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    b4 = GlobalAveragePooling2D()(x)
    # from (b_size, channels)->(b_size, 1, 1, channels)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', trainable=conv_trainable,
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(trainable=batch_normal_trainable, name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = Activation('relu')(b4)
    # upsample. have to use compat because of the option align_corners
    size_before = tf.shape(x)
    b4 = tf.compat.v1.image.resize(b4, size_before[1:3], method='bilinear', align_corners=True)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, trainable=conv_trainable, name='aspp0')(x)
    b0 = BatchNormalization(trainable=batch_normal_trainable, name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    if backbone == 'xception':
        # rate = 6 (12)
        b1 = SepConv_BN(x, 256, 'aspp1', batch_normal_trainable=batch_normal_trainable,
                        rate=atrous_rates[0], depth_activation=True, epsilon=1e-5,
                        conv_trainable=conv_trainable)
        # rate = 12 (24)
        b2 = SepConv_BN(x, 256, 'aspp2', batch_normal_trainable=batch_normal_trainable,
                        rate=atrous_rates[1], depth_activation=True, epsilon=1e-5,
                        conv_trainable=conv_trainable)
        # rate = 18 (36)
        b3 = SepConv_BN(x, 256, 'aspp3', batch_normal_trainable=batch_normal_trainable,
                        rate=atrous_rates[2], depth_activation=True, epsilon=1e-5,
                        conv_trainable=conv_trainable)

        # concatenate ASPP branches & project
        x = Concatenate()([b4, b0, b1, b2, b3])
    else:
        x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same', trainable=conv_trainable,
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(trainable=batch_normal_trainable,
                           name='concat_projection_BN', epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    # DeepLab v.3+ decoder

    if backbone == 'xception':
        # Feature projection
        # x4 (x2) block
        size_before2 = tf.shape(skip1)
        x = tf.compat.v1.image.resize(x, size_before2[1:3], method='bilinear', align_corners=True)

        dec_skip1 = Conv2D(48, (1, 1), padding='same', trainable=conv_trainable,
                           use_bias=False, name='feature_projection0')(skip1)
        dec_skip1 = BatchNormalization(trainable=batch_normal_trainable,
                                       name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
        dec_skip1 = Activation('relu')(dec_skip1)
        # print('x, skip', x.get_shape(), dec_skip1.get_shape())
        # exit()
        x = Concatenate()([x, dec_skip1])
        x = SepConv_BN(x, 256, 'decoder_conv0', batch_normal_trainable=batch_normal_trainable,
                       depth_activation=True, epsilon=1e-5, conv_trainable=conv_trainable)
        x = SepConv_BN(x, 256, 'decoder_conv1', batch_normal_trainable=batch_normal_trainable,
                       depth_activation=True, epsilon=1e-5, conv_trainable=conv_trainable)

    # you can use it with arbitary number of classes
    if (weights == 'pascal_voc' and num_class == 21) or \
            (weights == 'cityscapes' and num_class == 19):
        x = Conv2D(num_class, (1, 1), padding='same', trainable=conv_trainable, name='logits_semantic')(x)
    else:
        if weights in {'pascal_voc', 'cityscapes'} and with_pre_train_logits_semantic:
            def logits_semantic_kernel_initializer(shape, dtype=None, partition_info=None):
                channels_out = shape[3]
                filt_data = np.array(file_weights['/logits_semantic']['logits_semantic']['kernel:0'])
                para_shape = filt_data.shape
                list_filt_data = []
                for i in range(channels_out):
                    extra = filt_data[:, :, :, i % para_shape[3]]
                    extra = extra[:, :, :, np.newaxis]
                    list_filt_data.append(extra)
                filt_data = np.concatenate(list_filt_data, axis=3)
                return tf.convert_to_tensor(filt_data, dtype=dtype)

            def logits_semantic_bias_initializer(shape, dtype=None, partition_info=None):
                filt_data = np.array(file_weights['/logits_semantic']['logits_semantic']['bias:0'])
                para_shape = filt_data.shape
                list_filt_data = []
                for i in range(shape[0]):
                    extra = filt_data[i % para_shape[0]]
                    list_filt_data.append(extra)
                return tf.convert_to_tensor(list_filt_data, dtype=dtype)

            x = Conv2D(num_class, (1, 1), padding='same',
                       kernel_initializer=logits_semantic_kernel_initializer,
                       bias_initializer=logits_semantic_bias_initializer,
                       trainable=conv_trainable,
                       name='custom_logits_semantic')(x)
        else:
            x = Conv2D(num_class, (1, 1), padding='same', trainable=conv_trainable,
                       name='custom_logits_semantic')(x)
    size_before3 = tf.shape(inputs)
    x = tf.compat.v1.image.resize(x, size_before3[1:3], method='bilinear', align_corners=True)
    # Ensure that the model takes into account
    x = tf.keras.layers.Activation('softmax')(x)

    if experiment == 'gw_weight':
        x = tf.argmax(x, axis=-1)
        x = tf.cast(tf.expand_dims(x, axis=-1), dtype=tf.float32)
        patches_size = [1, 3, 3, 1]
        patches_stride = [1, 2, 2, 1]
        patches_rate = [1, 1, 1, 1]
        x = tf.image.extract_patches(x, sizes=patches_size,
                                     strides=patches_stride,
                                     rates=patches_rate, padding=str.upper('same'))
        x_unroll = tf.nn.depth_to_space(x, block_size=3)
        x = tf.concat([x_unroll, gw_weight], axis=-1)
        model = Model(inputs, x, name=name)
    else:
        model = Model(inputs, x, name=name)

    # load weights
    if weights is not None:
        model.load_weights(weights_path, by_name=True)

    return model


def Inception(x, fusion_mode, weights, file_weights, with_pre_train_entry_flow_conv1_1,
              batch_normal_trainable, conv_trainable, OS=16):
    if OS == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        atrous_rates = (12, 24, 36)
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        atrous_rates = (6, 12, 18)

    def conv1_1_kernel_initializer(shape, dtype=None, partition_info=None):
        channels_in = shape[2]
        filt_data = np.array(file_weights['/entry_flow_conv1_1']['entry_flow_conv1_1']['kernel:0'])
        para_shape = filt_data.shape  # (K, K, C_in, C_out)
        print('para_shape', para_shape)  # para_shape (3, 3, 3, 32)
        list_filt_data = []
        for i in range(channels_in):
            extra = filt_data[:, :, i % para_shape[2], :]
            extra = extra[:, :, np.newaxis, :]
            list_filt_data.append(extra)
        filt_data = np.concatenate(list_filt_data, axis=2)
        return tf.convert_to_tensor(filt_data, dtype=dtype)

    gw_weight = None
    if fusion_mode in {'bgr_hha', 'bgr_hha_xyz'} and weights is not None:
        if with_pre_train_entry_flow_conv1_1:
            x = Conv2D(32, (3, 3), strides=(2, 2), kernel_initializer=conv1_1_kernel_initializer,
                       trainable=conv_trainable,
                       name='custom_entry_flow_conv1_1', use_bias=False, padding='same')(x)

        else:
            x = Conv2D(32, (3, 3), strides=(2, 2), name='custom_entry_flow_conv1_1',
                       trainable=conv_trainable,
                       use_bias=False, padding='same')(x)
    elif fusion_mode == 'bgr_hha_gw':
        x, xyz, gw_weight = ilayers.GWConv(sizes=(3, 3), strides=(2, 2), rates=(1, 1),
                                           padding='same',
                                           delta=0.5, out_c=32, activation=None,
                                           name='gw_conv1_1')(x)

        x = Conv2D(32, (3, 3), strides=(3, 3), name='custom_entry_flow_conv1_1',
                   trainable=conv_trainable,
                   use_bias=False, padding='valid')(x)
    else:
        x = Conv2D(32, (3, 3), strides=(2, 2), trainable=conv_trainable,
                   name='entry_flow_conv1_1', use_bias=False, padding='same')(x)
    x = BatchNormalization(trainable=batch_normal_trainable, name='entry_flow_conv1_1_BN')(x)
    x = Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1, trainable=conv_trainable)
    x = BatchNormalization(trainable=batch_normal_trainable, name='entry_flow_conv1_2_BN')(x)
    x = Activation('relu')(x)

    x = _xception_block(x, [128, 128, 128], 'entry_flow_block1', batch_normal_trainable=batch_normal_trainable,
                        skip_connection_type='conv', stride=2, conv_trainable=conv_trainable,
                        depth_activation=False)
    x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2', batch_normal_trainable=batch_normal_trainable,
                               skip_connection_type='conv', stride=2, conv_trainable=conv_trainable,
                               depth_activation=False, return_skip=True)

    x = _xception_block(x, [728, 728, 728], 'entry_flow_block3', batch_normal_trainable=batch_normal_trainable,
                        skip_connection_type='conv', stride=entry_block3_stride, conv_trainable=conv_trainable,
                        depth_activation=False)
    for i in range(16):
        x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                            batch_normal_trainable=batch_normal_trainable,
                            skip_connection_type='sum', stride=1, rate=middle_block_rate,
                            depth_activation=False, conv_trainable=conv_trainable)

    x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1', batch_normal_trainable=batch_normal_trainable,
                        skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                        depth_activation=False, conv_trainable=conv_trainable)
    x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2', batch_normal_trainable=batch_normal_trainable,
                        skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                        depth_activation=True, conv_trainable=conv_trainable)
    return atrous_rates, x, skip1, gw_weight


def MobilenetV2(x, alpha, batch_normal_trainable, conv_trainable):
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
               kernel_size=3, trainable=conv_trainable,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(x)
    x = BatchNormalization(trainable=batch_normal_trainable,
                           epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
    x = Activation(relu6, name='Conv_Relu6')(x)

    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1, batch_normal_trainable=batch_normal_trainable,
                            expansion=1, block_id=0, skip_connection=False, conv_trainable=conv_trainable)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=1, skip_connection=False, conv_trainable=conv_trainable)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=2, skip_connection=True, conv_trainable=conv_trainable)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=3, skip_connection=False, conv_trainable=conv_trainable)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=4, skip_connection=True, conv_trainable=conv_trainable)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=5, skip_connection=True, conv_trainable=conv_trainable)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, batch_normal_trainable=batch_normal_trainable,  # 1!
                            expansion=6, block_id=6, skip_connection=False, conv_trainable=conv_trainable)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=7, skip_connection=True, conv_trainable=conv_trainable)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=8, skip_connection=True, conv_trainable=conv_trainable)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=9, skip_connection=True, conv_trainable=conv_trainable)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=10, skip_connection=False, conv_trainable=conv_trainable)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=11, skip_connection=True, conv_trainable=conv_trainable)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2, batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=12, skip_connection=True, conv_trainable=conv_trainable)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,
                            batch_normal_trainable=batch_normal_trainable,  # 1!
                            expansion=6, block_id=13, skip_connection=False, conv_trainable=conv_trainable)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=14, skip_connection=True, conv_trainable=conv_trainable)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=15, skip_connection=True, conv_trainable=conv_trainable)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            batch_normal_trainable=batch_normal_trainable,
                            expansion=6, block_id=16, skip_connection=False, conv_trainable=conv_trainable)
    return x


def SepConv_BN(x, filters, prefix, batch_normal_trainable, conv_trainable,
               stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('relu')(x)
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        trainable=conv_trainable,
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
    x = BatchNormalization(trainable=batch_normal_trainable,
                           name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same', trainable=conv_trainable,
               use_bias=False, name=prefix + '_pointwise')(x)
    x = BatchNormalization(trainable=batch_normal_trainable,
                           name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = Activation('relu')(x)

    return x


def _conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1, trainable=True):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      trainable=trainable,
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      trainable=trainable,
                      name=prefix)(x)


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride, batch_normal_trainable,
                    conv_trainable, rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              batch_normal_trainable=batch_normal_trainable,
                              conv_trainable=conv_trainable,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1, trainable=conv_trainable,
                                stride=stride)
        shortcut = BatchNormalization(trainable=batch_normal_trainable,
                                      name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def relu6(x):
    return relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection,
                        batch_normal_trainable, conv_trainable, rate=1):
    in_channels = inputs.shape.as_list()[-1]  # inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None, trainable=conv_trainable,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(trainable=batch_normal_trainable, epsilon=1e-3, momentum=0.999,
                               name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, trainable=conv_trainable,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(trainable=batch_normal_trainable, epsilon=1e-3, momentum=0.999,
                           name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters, trainable=conv_trainable,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(trainable=batch_normal_trainable, epsilon=1e-3, momentum=0.999,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x
