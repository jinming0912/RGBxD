from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import random
import numpy as np
import tensorflow as tf


def instance_normalization(data, is_training, name, reuse=None):
    return tf.contrib.layers.instance_norm(inputs=data, center=True, scale=True, epsilon=1e-06, activation_fn=None,
                                           param_initializers=None, reuse=None, variables_collections=None,
                                           outputs_collections=None, trainable=True, scope=None)


def batch_normalization(data, is_training, name, reuse=None):
    return tf.layers.batch_normalization(data, momentum=0.99, training=is_training,
                                         beta_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                         gamma_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                         reuse=reuse, name=name)


def separable_conv2d(input, output, name, is_training, kernel_size, depth_multiplier=1,
                     reuse=None, with_bn=True, activation=tf.nn.elu):
    conv2d = tf.layers.separable_conv2d(input, output, kernel_size=kernel_size, strides=(1, 1), padding='VALID',
                                        activation=activation,
                                        depth_multiplier=depth_multiplier,
                                        depthwise_initializer=tf.glorot_normal_initializer(),
                                        pointwise_initializer=tf.glorot_normal_initializer(),
                                        depthwise_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                        pointwise_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                        reuse=reuse, name=name, use_bias=not with_bn)
    return batch_normalization(conv2d, is_training, name + '_bn', reuse) if with_bn else conv2d


def depthwise_conv2d(input, depth_multiplier, name, is_training, kernel_size,
                     reuse=None, with_bn=True, activation=tf.nn.elu):
    conv2d = tf.contrib.layers.separable_conv2d(input, num_outputs=None, kernel_size=kernel_size, padding='VALID',
                                                activation_fn=activation,
                                                depth_multiplier=depth_multiplier,
                                                weights_initializer=tf.glorot_normal_initializer(),
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                                biases_initializer=None if with_bn else tf.zeros_initializer(),
                                                biases_regularizer=None if with_bn else tf.contrib.layers.l2_regularizer(
                                                    scale=1.0),
                                                reuse=reuse, scope=name)
    return batch_normalization(conv2d, is_training, name + '_bn', reuse) if with_bn else conv2d



def conv2d(input, output, name, is_training, kernel_size,
           reuse=None, with_bn=True, activation=tf.nn.elu):
    conv2d = tf.layers.conv2d(input, output, kernel_size=kernel_size, strides=(1, 1), padding='VALID',
                              activation=activation,
                              kernel_initializer=tf.glorot_normal_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                              reuse=reuse, name=name, use_bias=not with_bn)
    return batch_normalization(conv2d, is_training, name + '_bn', reuse) if with_bn else conv2d


def deconv2d(input, output, name, is_training, kernel_size, stride=(1, 1),
             reuse=None, with_bn=True, activation=tf.nn.elu, padding='valid'):
    deconv2d = tf.layers.conv2d_transpose(input, output, kernel_size=kernel_size, strides=stride, padding=padding,
                                          activation=activation,
                                          kernel_initializer=tf.glorot_normal_initializer(),
                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                                          reuse=reuse, name=name, use_bias=not with_bn)
    return batch_normalization(deconv2d, is_training, name + '_bn', reuse) if with_bn else deconv2d


def dense(input, output, name, is_training, reuse=None, with_bn=True, activation=tf.nn.elu):
    dense = tf.layers.dense(input, units=output, activation=activation,
                            kernel_initializer=tf.glorot_normal_initializer(),
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=1.0),
                            reuse=reuse, name=name, use_bias=not with_bn)
    return batch_normalization(dense, is_training, name + '_bn', reuse) if with_bn else dense


def center_pooling(value, ksize, strides, padding, name=None):
    with tf.name_scope(name):
        patches_d = tf.compat.v1.extract_image_patches(images=value, ksizes=ksize, strides=strides,
                                             rates=[1, 1, 1, 1],
                                             padding=padding)

        # patches_d_center = patches_d[:, :, :, ksize[1] * ksize[2] // 2]
        # return patches_d_center[:, :, :, tf.newaxis]
        patches_d = tf.reshape(patches_d, [-1] + patches_d.get_shape().as_list()[1:3] + [4, 3])
        patches_d_center = patches_d[:, :, :, ksize[1] * ksize[2] // 2, :]
        return patches_d_center


def sum_pooling(value, pool_size, strides, padding, name=None):
    with tf.name_scope(name):
        avg_pool = tf.keras.layers.AvgPool2D(pool_size=pool_size, strides=strides, padding=padding)(value)
        return avg_pool*pool_size[0]*pool_size[1]
        # return tf.keras.layers.Conv2D(filters=value.get_shape()[-1], kernel_size=pool_size, strides=strides,
        #                               padding=padding, use_bias=False, kernel_initializer=tf.initializers.ones,
        #                               trainable=False)(value)


def unpool_2d(pool,
              ind,
              out_size,
              stride=[1, 2, 2, 1],
              scope='unpool_2d'):
    """Adds a 2D unpooling op.
    Unpooling layer after max_pool_with_argmax.
        Args:
           pool:        max pooled output tensor
           ind:         argmax indices
           stride:      stride is the same as for the pool
        Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape = tf.shape(pool)
        channels = pool.get_shape().as_list()[-1]
        # output_shape = [input_shape[0], input_shape[1] * stride[1], input_shape[2] * stride[2], input_shape[3]]
        output_shape = [input_shape[0], out_size[0], out_size[1], channels]

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype),
                                 shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b1 = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b1, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, output_shape)

        set_input_shape = pool.get_shape()
        # set_output_shape = [set_input_shape[0], set_input_shape[1] * stride[1], set_input_shape[2] * stride[2],
        #                     set_input_shape[3]]
        set_output_shape = [set_input_shape[0], out_size[0], out_size[1], set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret
