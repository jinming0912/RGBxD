import logging
import tensorflow as tf
import layers
from tensorflow.python.keras.engine import Layer
import numpy as np


class GWConv(Layer):
    def __init__(self,
                 sizes,
                 strides,
                 rates,
                 padding,
                 delta,
                 out_c,
                 activation,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(GWConv, self).__init__(
            trainable=trainable,
            name=name,
            **kwargs)

        # self.delta = delta

        self.delta = tf.Variable(delta, name=name + 'delta', trainable=trainable)
        self.sizes = sizes
        self.strides = strides
        self.rates = rates
        self.padding = padding
        self.activation = activation
        self.out_c = out_c

        self.kernel_h, self.kernel_w = (self.sizes, self.sizes) if isinstance(self.sizes, int) else self.sizes
        print("self.kernel_h, self.kernel_w", self.kernel_h, self.kernel_w)

        self.gw_conv_bn1 = layers.IConv(self.kernel_h * self.kernel_w * 2, self.sizes,
                                         strides=self.sizes,
                                         dilation_rate=1,
                                         padding='valid', name=name + 'conv_bn1', activation='elu',
                                         use_batch_normal=True)
        self.gw_conv_bn2 = layers.IConv(self.kernel_h * self.kernel_w, 1,
                                         strides=1,
                                         dilation_rate=1,
                                         padding='valid', name=name + 'conv_bn2',
                                         activation='sigmoid',
                                         use_batch_normal=False)
        self.feature_conv = layers.IConv(self.out_c, self.sizes,
                                         strides=self.sizes,
                                         dilation_rate=1,
                                         padding='valid', name=name + 'feature_conv',
                                         activation=self.activation,
                                         use_batch_normal=False)

        # self.xyz_conv_bn3 = layers.IConv(self.kernel_h * self.kernel_w * 3, 1,
        #                                  strides=1,
        #                                  dilation_rate=1,
        #                                  padding='VALID', name=name + 'conv_bn3', activation='elu',
        #                                  use_batch_normal=True)

    def call(self, inputs, **kwargs):
        C_in = inputs.get_shape().as_list()[3]
        xyz_c = 3
        features, xyz = tf.split(inputs, [C_in - xyz_c, xyz_c], axis=-1)

        rgb_features, other_features = tf.split(features, [3, C_in - xyz_c - 3], axis=-1)
        # rgb_features = features

        print('features', features.get_shape())
        print('xyz', xyz.get_shape())

        patches_size = [1, self.sizes[0], self.sizes[1], 1]
        patches_stride = [1, self.strides[0], self.strides[1], 1]
        patches_rate = [1, self.rates[0], self.rates[1], 1]

        rgb_features_patches = tf.image.extract_patches(rgb_features, sizes=patches_size,
                                                        strides=patches_stride,
                                                        rates=patches_rate, padding=str.upper(self.padding))
        print('rgb_features_patches', rgb_features_patches.get_shape())
        # (N, H_out, W_out, kernel_h*kernel_w*C_in)

        rgb_features_unroll = tf.nn.depth_to_space(rgb_features_patches, block_size=self.sizes[0])
        print('rgb_features_unroll', rgb_features_unroll.get_shape())
        # (N, H_out*kernel_h, W_out*kernel_w, C_in)

        xyz_patches = tf.image.extract_patches(xyz, sizes=patches_size, strides=patches_stride,
                                               rates=patches_rate, padding=str.upper(self.padding))
        print('xyz_patches', xyz_patches.get_shape())
        # (N, H_out, W_out, kernel_h*kernel_w*3)

        ####################################################
        # make xyzs relative to the center point
        N = -1
        H_out = tf.shape(xyz_patches)[1]
        W_out = tf.shape(xyz_patches)[2]
        P3 = tf.shape(xyz_patches)[3]
        print('N, H_out, W_out, P3', N, H_out, W_out, P3)
        C = xyz.get_shape().as_list()[3]
        print('C', C)

        xyz_patches_reshape = tf.reshape(xyz_patches, (N, H_out, W_out, self.kernel_h, self.kernel_w, C))
        print('xyz_patches_reshape', xyz_patches_reshape.get_shape())
        xyz_patches_center = xyz_patches_reshape[:, :, :, self.kernel_h // 2, self.kernel_w // 2, :]
        print('xyz_patches_center', xyz_patches_center.get_shape())
        # (N, H_out, W_out, 3)
        xyz_patches_center_5D = tf.expand_dims(xyz_patches_center, axis=3)
        xyz_patches_reshape = tf.reshape(xyz_patches_reshape, (N, H_out, W_out,
                                                               self.kernel_h * self.kernel_w, C))
        xyz_patches = xyz_patches_reshape - xyz_patches_center_5D
        print('xyz_patches', xyz_patches.get_shape())
        # (N, H_out, W_out, kernel_h*kernel_w, 3)
        xyz_patches = tf.reshape(xyz_patches, (N, H_out, W_out, self.kernel_h * self.kernel_w * C))

        ####################################################

        xyz_unroll = tf.nn.depth_to_space(xyz_patches, block_size=self.sizes[0])
        print('xyz_unroll', xyz_unroll.get_shape())
        # (N, H_out*kernel_h, W_out*kernel_w, C)
        xyz_square = tf.math.square(xyz_unroll)

        xyz_input = tf.concat([xyz_unroll, xyz_square], axis=-1)
        print('xyz_input', xyz_input.get_shape())
        # (N, H_out*kernel_h, W_out*kernel_w, 2*C)

        gw_conv1 = self.gw_conv_bn1(xyz_input)
        print('gw_conv1', gw_conv1.get_shape())
        # (N, H_ou, W_out, 2*kernel_h*kernel_w)
        gw_conv2 = self.gw_conv_bn2(gw_conv1)
        print('gw_conv2', gw_conv2.get_shape())
        # (N, H_out, W_out, kernel_h*kernel_w)

        gw_weight = tf.nn.depth_to_space(gw_conv2, block_size=self.sizes[0]) + self.delta
        print('gw_weight', gw_weight.get_shape())
        # (N, H_out*kernel_h, W_out*kernel_w, 1)
        rgb_features_weighted = tf.multiply(rgb_features_unroll, gw_weight)
        print('rgb_features_weighted', rgb_features_weighted.get_shape())
        # (N, H_out*kernel_h, W_out*kernel_w, C)

        other_features_patches = tf.image.extract_patches(other_features, sizes=patches_size,
                                                          strides=patches_stride,
                                                          rates=patches_rate, padding=str.upper(self.padding))
        print('other_features_patches', other_features_patches.get_shape())
        other_features_unroll = tf.nn.depth_to_space(other_features_patches, block_size=self.sizes[0])
        print('other_features_unroll', other_features_unroll.get_shape())

        features_weighted = tf.concat((rgb_features_weighted, other_features_unroll), axis=-1)
        return features_weighted, xyz_patches_center, tf.concat([gw_weight, rgb_features_unroll], axis=-1)
