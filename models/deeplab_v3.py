import logging
import os

import tensorflow as tf
import layers as ilayers
from tensorflow.python.keras import layers as klayers
from models.base_model import BaseModel


class DeeplabV3(BaseModel):
    r"""DeepLabV3

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.
    """

    def __init__(self, input_shape,
                 batch_size,
                 num_class,
                 fusion_mode,
                 output_stride,
                 weights=None,
                 name='DeeplabV3',
                 backbone='resnet50',
                 backbone_weights=None,
                 dilated=True,
                 multi_grid=False,
                 multi_dilation=None,
                 norm_layer=None,
                 norm_kwargs=None,
                 conv_trainable=True,
                 **kwargs):

        self.fusion_mode = fusion_mode
        if self.fusion_mode in ('bgr', 'hha'):
            backbone_input_shape = [batch_size, input_shape[0], input_shape[1], 3]
        elif self.fusion_mode == 'bgr_hha':
            backbone_input_shape = [batch_size, input_shape[0], input_shape[1], 6]
        elif self.fusion_mode in ('bgr_hha_xyz', 'bgr_hha_gw'):
            backbone_input_shape = [batch_size, input_shape[0], input_shape[1], 9]
        else:
            logging.error("Unknown fusion mode.")
            return
        super(DeeplabV3, self).__init__(backbone,
                                        backbone_input_shape,
                                        fusion_mode=fusion_mode,
                                        dilated=dilated,
                                        multi_grid=multi_grid,
                                        backbone_weights=backbone_weights,
                                        multi_dilation=multi_dilation,
                                        conv_trainable=conv_trainable,
                                        name=name)
        self.output_stride = output_stride
        self.input_layer = klayers.InputLayer(batch_input_shape=[batch_size] + input_shape, dtype=tf.float32)
        self.split_inputs = klayers.Lambda(lambda x: tf.split(axis=-1, num_or_size_splits=[3, 3, 1, 3], value=x))

        self.concat_bgr_hha = klayers.Concatenate()

        self.head = DeepLabHead(num_class, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                conv_trainable=True, **kwargs)

    def call(self, inputs, training=None, mask=None):
        print('inputs', inputs)
        # tf.print('inputs', inputs)
        x = self.input_layer(inputs)
        bgr, hha, dep, xyz = self.split_inputs(x)
        if self.fusion_mode == 'bgr_hha':
            x = self.concat_bgr_hha([bgr, hha])
        elif self.fusion_mode == 'bgr':
            x = bgr
        elif self.fusion_mode == 'hha':
            x = hha
        elif self.fusion_mode in ('bgr_hha_gw', 'bgr_hha_xyz'):
            x = tf.concat([bgr, hha, xyz], axis=-1)
        else:
            x = None
        x = super(DeeplabV3, self).call(x)
        x = self.head(x)
        inputs_shape = tf.shape(inputs)
        x = tf.compat.v1.image.resize(x, inputs_shape[1:3], method='bilinear', align_corners=True)
        return x


class DeepLabHead(klayers.Layer):
    def __init__(self, nclass, norm_layer=None, norm_kwargs=None, conv_trainable=True, **kwargs):
        super(DeepLabHead, self).__init__()
        self.aspp = ASPP([12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                         conv_trainable=conv_trainable, **kwargs)
        self.block = tf.keras.Sequential([
            klayers.Conv2D(256, kernel_size=3, padding='same', kernel_initializer='he_uniform', use_bias=False,
                           trainable=conv_trainable),
            norm_layer(**({} if norm_kwargs is None else norm_kwargs)),
            klayers.ReLU(),
            klayers.Dropout(0.1),
            klayers.Conv2D(nclass, kernel_initializer='he_uniform', kernel_size=1, trainable=conv_trainable)
        ])

    def call(self, x, **kwargs):
        x = self.aspp(x)
        x = self.block(x)
        return x


class ASPPConv(klayers.Layer):
    def __init__(self, out_channels, atrous_rate, norm_layer, norm_kwargs, conv_trainable=True):
        super(ASPPConv, self).__init__()
        self.conv = klayers.Conv2D(out_channels, kernel_size=3, padding='same', kernel_initializer='he_uniform',
                                   dilation_rate=atrous_rate, use_bias=False, trainable=conv_trainable)
        self.bn = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.relu = klayers.ReLU()

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPPooling(klayers.Layer):
    def __init__(self, out_channels, norm_layer, norm_kwargs, conv_trainable=True, **kwargs):
        super(ASPPPooling, self).__init__()
        self.gap = tf.keras.Sequential([
            klayers.GlobalAveragePooling2D(),
            klayers.Lambda(lambda x: tf.keras.backend.expand_dims(x, 1)),
            klayers.Lambda(lambda x: tf.keras.backend.expand_dims(x, 1)),
            klayers.Conv2D(out_channels, kernel_size=1, kernel_initializer='he_uniform', use_bias=False,
                           trainable=conv_trainable),
            norm_layer(**({} if norm_kwargs is None else norm_kwargs)),
            klayers.ReLU()
        ])

    def call(self, x, **kwargs):
        pool = self.gap(x)
        # upsample. have to use compat because of the option align_corners
        size_before = tf.shape(x)
        pool = tf.compat.v1.image.resize(pool, size_before[1:3], method='bilinear', align_corners=True)

        return pool


class ASPP(klayers.Layer):
    def __init__(self, atrous_rates, norm_layer, norm_kwargs, conv_trainable=True, **kwargs):
        super(ASPP, self).__init__()
        out_channels = 256
        self.b0 = tf.keras.Sequential([
            klayers.Conv2D(out_channels, kernel_size=1, kernel_initializer='he_uniform', use_bias=False,
                           trainable=conv_trainable),
            norm_layer(**({} if norm_kwargs is None else norm_kwargs)),
            klayers.ReLU()
        ])

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = ASPPConv(out_channels, rate1, norm_layer, norm_kwargs, conv_trainable=conv_trainable)
        self.b2 = ASPPConv(out_channels, rate2, norm_layer, norm_kwargs, conv_trainable=conv_trainable)
        self.b3 = ASPPConv(out_channels, rate3, norm_layer, norm_kwargs, conv_trainable=conv_trainable)
        self.b4 = ASPPPooling(out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                              conv_trainable=conv_trainable)
        self.concat = klayers.Concatenate()

        self.project = tf.keras.Sequential([
            klayers.Conv2D(out_channels, kernel_size=1, kernel_initializer='he_uniform', use_bias=False,
                           trainable=conv_trainable),
            norm_layer(**({} if norm_kwargs is None else norm_kwargs)),
            klayers.ReLU(),
            klayers.Dropout(0.5)
        ])

    def call(self, x, **kwargs):
        # tf.print('x in aspp', tf.shape(x))
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = self.concat([feat1, feat2, feat3, feat4, feat5])
        x = self.project(x)
        return x
