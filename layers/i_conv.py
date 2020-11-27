import os

import numpy as np
import tensorflow as tf
import logging

from layers.i_layer import ILayer


class IConv(ILayer):
    _data_dict_vgg16 = None

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='same',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='relu',
                 use_bias=True,
                 # kernel_initializer='vgg16',
                 kernel_initializer='glorot_uniform',
                 # bias_initializer='vgg16',
                 bias_initializer='zeros',
                 use_batch_normal=True,
                 trainable=True,
                 name='iconv',
                 **kwargs):
        super(IConv, self).__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            trainable=trainable,
            name=name,
            **kwargs)
        if kernel_initializer == 'vgg16':
            kernel_initializer = self.__get_filter_vgg16
            if IConv._data_dict_vgg16 is None:
                logging.info("Load vgg16 pre-train parameters.")
                IConv._data_dict_vgg16 = np.load(os.path.join(os.environ['HOME'], "Models/vgg16.npy"),
                                                 encoding='latin1', allow_pickle=True).item()

        if bias_initializer == 'vgg16':
            bias_initializer = self.__get_bias_vgg16
            if IConv._data_dict_vgg16 is None:
                logging.info("Load vgg16 pre-train parameters.")
                IConv._data_dict_vgg16 = np.load(os.path.join(os.environ['HOME'], "Models/vgg16.npy"),
                                                 encoding='latin1', allow_pickle=True).item()

        if '/' in self.name:
            self.para_name = self.name.split('/')[-1]
        else:
            self.para_name = self.name
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           data_format=data_format,
                                           dilation_rate=dilation_rate,
                                           activation=None,
                                           use_bias=use_bias,
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           trainable=trainable,
                                           name=name + '_conv',
                                           kernel_regularizer=None,
                                           bias_regularizer=None,
                                           activity_regularizer=None,
                                           kernel_constraint=None,
                                           bias_constraint=None,
                                           **kwargs)
        self.use_batch_normal = use_batch_normal
        self.activation = activation
        self.batch_normal = None

        if self.use_batch_normal:
            self.batch_normal = tf.keras.layers.BatchNormalization(name=self.name + '_bn')
        if self.activation is not None:
            self.activation = tf.keras.layers.Activation(self.activation)
        return

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        # out_shape = self.conv.compute_output_shape(inputs.get_shape())
        # x.set_shape(out_shape)
        if self.use_batch_normal:
            x = self.batch_normal(x, training=training)
        if self.activation is not None:
            x = self.activation(x)

        return x

    def __get_filter_vgg16(self, shape, dtype=None, partition_info=None):
        # (3, 3, 64, 128)
        logging.debug('__get_filter_vgg16 shape: %s' % str(shape))
        channels_in = shape[2]
        channels_out = shape[3]
        kernel_size = shape[0:2]
        filt_data = IConv._data_dict_vgg16[self.para_name][0]
        para_shape = filt_data.shape
        if len(para_shape) == 2:
            filt_data = np.reshape(filt_data, (1, 1, para_shape[0], para_shape[1]))
            para_shape = filt_data.shape

        if channels_out != para_shape[3]:
            logging.info("%s pre-train parameters channel_out resize." % self.name)
            list_filt_data = []
            for i in range(channels_out):
                extra = filt_data[:, :, :, i % para_shape[3]]
                extra = extra[:, :, :, np.newaxis]
                list_filt_data.append(extra)
            filt_data = np.concatenate(list_filt_data, axis=3)
            para_shape = filt_data.shape

        if channels_in != para_shape[2]:
            logging.info("%s pre-train parameters channel_in resize." % self.name)
            list_filt_data = []
            for i in range(channels_in):
                extra = filt_data[:, :, i % para_shape[2], :]
                extra = extra[:, :, np.newaxis, :]
                list_filt_data.append(extra)
            filt_data = np.concatenate(list_filt_data, axis=2)
            para_shape = filt_data.shape

        if list(kernel_size) != list(para_shape[0:2]):
            logging.info('%s pre-train parameters kernel resize.', self.name)
            filt_data = np.reshape(filt_data, (para_shape[0] * para_shape[1], para_shape[2], para_shape[3]))
            list_filt_data = []
            for i in range(kernel_size[0] * kernel_size[1]):
                extra = filt_data[i % (para_shape[0] * para_shape[1]), :, :]
                extra = extra[np.newaxis, :, :]
                list_filt_data.append(extra)
            filt_data = np.concatenate(list_filt_data, axis=0)
            filt_data = np.reshape(filt_data, list(kernel_size) + list(filt_data.shape[-2:]))

        return tf.convert_to_tensor(filt_data, dtype=dtype)

    def __get_bias_vgg16(self, shape, dtype=tf.float32, partition_info=None):
        logging.debug('__get_bias_vgg16 shape: %s' % str(shape))
        bias_data = IConv._data_dict_vgg16[self.para_name][1]
        if shape[0] <= bias_data.shape[0]:
            bias_data = bias_data[0:shape[0]]
        else:
            logging.error("Unsupport bias shape.")
            return None
        biases = tf.convert_to_tensor(bias_data, dtype=dtype)
        return biases
