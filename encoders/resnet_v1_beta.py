"""Dilated ResNet"""
import os

import tensorflow as tf
from tensorflow.python.keras import layers as klayers
import layers as ilayers

WEIGHTS_PATH = ('url/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('url/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


def identity_block(input_tensor, kernel_size, filters, stage, block, dilation=1, conv_trainable=True):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = klayers.Conv2D(filters1, (1, 1),
                       kernel_initializer='he_uniform',
                       trainable=conv_trainable,
                       name=conv_name_base + '2a')(input_tensor)
    x = klayers.BatchNormalizationV2(name=bn_name_base + '2a')(x)
    x = klayers.Activation('relu')(x)

    x = klayers.Conv2D(filters2, kernel_size, dilation_rate=dilation,
                       padding='same',
                       kernel_initializer='he_uniform',
                       trainable=conv_trainable,
                       name=conv_name_base + '2b')(x)
    x = klayers.BatchNormalizationV2(name=bn_name_base + '2b')(x)
    x = klayers.Activation('relu')(x)

    x = klayers.Conv2D(filters3, (1, 1),
                       kernel_initializer='he_uniform',
                       trainable=conv_trainable,
                       name=conv_name_base + '2c')(x)
    x = klayers.BatchNormalizationV2(name=bn_name_base + '2c')(x)

    x = klayers.add([x, input_tensor])
    x = klayers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=2,
               dilation=1,
               conv_trainable=True):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = klayers.Conv2D(filters1, (1, 1), strides=strides,
                       kernel_initializer='he_uniform',
                       trainable=conv_trainable,
                       name=conv_name_base + '2a')(input_tensor)
    x = klayers.BatchNormalizationV2(name=bn_name_base + '2a')(x)
    x = klayers.Activation('relu')(x)

    x = klayers.Conv2D(filters2, kernel_size, padding='same', dilation_rate=dilation,
                       kernel_initializer='he_uniform',
                       trainable=conv_trainable,
                       name=conv_name_base + '2b')(x)
    x = klayers.BatchNormalizationV2(name=bn_name_base + '2b')(x)
    x = klayers.Activation('relu')(x)

    x = klayers.Conv2D(filters3, (1, 1),
                       kernel_initializer='he_uniform',
                       trainable=conv_trainable,
                       name=conv_name_base + '2c')(x)
    x = klayers.BatchNormalizationV2(name=bn_name_base + '2c')(x)

    shortcut = klayers.Conv2D(filters3, (1, 1), strides=strides,
                              kernel_initializer='he_uniform',
                              trainable=conv_trainable,
                              name=conv_name_base + '1')(input_tensor)
    shortcut = klayers.BatchNormalizationV2(name=bn_name_base + '1')(shortcut)

    x = klayers.add([x, shortcut])
    x = klayers.Activation('relu')(x)
    return x


def resnet_block(x, base_filters, num_units, stage, dilation=1, stride=2,
                 multi_grid=False, multi_dilation=None, conv_trainable=True):
    expansion = 4
    if multi_grid:
        x = conv_block(x, 3, [base_filters, base_filters, base_filters * expansion],
                       stage=stage, block='a', strides=stride, dilation=multi_dilation[0],
                       conv_trainable=conv_trainable)
    else:
        if dilation == 1 or dilation == 2:
            x = conv_block(x, 3, [base_filters, base_filters, base_filters * expansion],
                           stage=stage, block='a', strides=stride, dilation=1,
                           conv_trainable=conv_trainable)
        elif dilation == 4:
            x = conv_block(x, 3, [base_filters, base_filters, base_filters * expansion],
                           stage=stage, block='a', strides=stride, dilation=2,
                           conv_trainable=conv_trainable)
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

    if multi_grid:
        div = len(multi_dilation)
        for i in range(1, num_units):
            x = identity_block(x, 3, [base_filters, base_filters, base_filters * expansion],
                               stage=stage, block=chr(i + 97), dilation=multi_dilation[i % div],
                               conv_trainable=conv_trainable)
    else:
        for i in range(1, num_units):
            x = identity_block(x, 3, [base_filters, base_filters, base_filters * expansion],
                               stage=stage, block=chr(i + 97), dilation=dilation,
                               conv_trainable=conv_trainable)

    return x


def ResNet50Beta(batch_input_shape,
                 include_top=True,
                 weights='imagenet',
                 fusion_mode='bgr_hha',
                 classes=1000,
                 dilated=True,
                 multi_grid=False,
                 multi_dilation=None,
                 conv_trainable=True):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional block.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    img_input = klayers.Input(shape=batch_input_shape[1:4], batch_size=batch_input_shape[0])

    if fusion_mode == 'bgr_hha_gw':
        x, xyz = ilayers.GWConv(sizes=(3, 3), strides=(2, 2), rates=(1, 1),
                                padding='same',
                                delta=0.5, out_c=64, activation=None,
                                name='gw_conv1_1')(img_input)

        x = klayers.Conv2D(64, (3, 3), strides=(3, 3), name='custom_entry_flow_conv1_1',
                           trainable=conv_trainable,
                           use_bias=False, padding='valid')(x)
    else:
        # x = klayers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = klayers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer='he_uniform',
                           trainable=True,
                           name='conv1' if batch_input_shape[-1] == 3 else 'custom_conv1')(img_input)
    x = klayers.BatchNormalizationV2(name='bn_conv1')(x)
    x = klayers.Activation('relu')(x)
    x = klayers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = klayers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_block(x, num_units=3, base_filters=64, stage=2, stride=1, conv_trainable=conv_trainable)
    x = resnet_block(x, num_units=4, base_filters=128, stage=3, stride=2, conv_trainable=conv_trainable)
    if dilated:
        if multi_grid:
            x = resnet_block(x, num_units=6, base_filters=256, stage=4, dilation=2, stride=1,
                             conv_trainable=conv_trainable)
            x = resnet_block(x, num_units=3, base_filters=512, stage=5, dilation=4, stride=1,
                             multi_grid=multi_grid, multi_dilation=multi_dilation, conv_trainable=conv_trainable)
        else:
            x = resnet_block(x, num_units=6, base_filters=256, stage=4, dilation=2, stride=1,
                             conv_trainable=conv_trainable)
            x = resnet_block(x, num_units=3, base_filters=512, stage=5, dilation=4, stride=1,
                             conv_trainable=conv_trainable)
    else:
        x = resnet_block(x, num_units=6, base_filters=256, stage=4, stride=2, conv_trainable=conv_trainable)
        x = resnet_block(x, num_units=3, base_filters=512, stage=5, stride=2, conv_trainable=conv_trainable)

    if include_top:
        x = klayers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = klayers.Dense(classes, activation='softmax', name='fc1000')(x)

    # Create model.
    model = tf.keras.models.Model(img_input, x, name='resnet50_beta')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = tf.keras.utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = tf.keras.utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path, by_name=True)
    elif weights is not None:
        model.load_weights(weights, by_name=True)

    return model
