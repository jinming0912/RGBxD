import logging
import tensorflow as tf

import encoders
import layers as ilayers


class UNet(tf.keras.Model):
    def __init__(self,
                 num_class,
                 fusion_mode,
                 name='UNet',
                 backbone='mobile_net_v2',
                 backbone_initializer='imagenet'):
        super(UNet, self).__init__(name=name)
        self.num_class = num_class
        self.fusion_mode = fusion_mode
        self.backbone = backbone
        self.backbone_initializer = backbone_initializer

    def build(self, input_shape, output_channels=None):
        print('input_shape[1:4]', input_shape[1:4])
        img_shape = list(input_shape[1:3])
        if self.fusion_mode == 'bgr':
            input_shape = img_shape + [3]
        elif self.fusion_mode == 'bgr_hha':
            input_shape = img_shape + [6]
        elif self.fusion_mode in ('bgr_hha_xyz', 'bgr_hha_gw'):
            input_shape = img_shape + [9]
        else:
            input_shape = None
            logging.error('Unknown fusion mode.')
        if self.backbone == 'mobile_net_v2':
            base_model = encoders.MobileNetV2(fusion_mode=self.fusion_mode,
                                              input_shape=input_shape,
                                              include_top=False,
                                              weights=self.backbone_initializer)
        else:
            base_model = None
            logging.error("Unknown backbone.")

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        self.down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

        self.up_stack = [
            self.upsample(512, 3),  # 4x4 -> 8x8
            self.upsample(256, 3),  # 8x8 -> 16x16
            self.upsample(128, 3),  # 16x16 -> 32x32
            self.upsample(64, 3),  # 32x32 -> 64x64
        ]
        # This is the last layer of the model

        if self.fusion_mode == 'bgr_hha_gw':
            self.last_up = self.upsample(64, 3)
            self.last_gw = ilayers.GWConv(sizes=(3, 3), strides=(1, 1), rates=(1, 1),
                                          padding='same',
                                          delta=0.5, out_c=64, activation=None,
                                          name='gw_deconv1_1')
            self.last = tf.keras.layers.Conv2D(
                self.num_class, 3, strides=3,
                padding='valid', activation='softmax')

            # self.last = tf.keras.layers.Conv2DTranspose(
            #     self.num_class, 3, strides=2,
            #     padding='same', activation='softmax')  # 64x64 -> 128x128
        else:
            self.last = tf.keras.layers.Conv2DTranspose(
                self.num_class, 3, strides=2,
                padding='same', activation='softmax')  # 64x64 -> 128x128

    def call(self, inputs, **kwargs):
        if inputs is None:
            inputs = tf.keras.layers.Input(shape=self.input_shape)
        bgr, hha, dep, xyz = tf.split(axis=-1, num_or_size_splits=[3, 3, 1, 3], value=inputs)
        if self.fusion_mode == 'bgr':
            x = bgr
        elif self.fusion_mode == 'bgr_hha':
            x = tf.concat([bgr, hha], axis=-1)
        elif self.fusion_mode in ('bgr_hha_xyz', 'bgr_hha_gw'):
            x = tf.concat([bgr, hha, xyz], axis=-1)
        else:
            x = None
            logging.error('Unknown fusion mode.')

        # Downsampling through the model
        skips = self.down_stack(x, **kwargs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        if self.fusion_mode == 'bgr_hha_gw':
            x = self.last_up(x)
            x = tf.concat([x, xyz], axis=-1)
            x, xyz, _ = self.last_gw(x)
        x = self.last(x)
        return x

    def upsample(self, filters, size, norm_type='batchnorm', apply_dropout=False):
        """Upsamples an input.
        Conv2DTranspose => Batchnorm => Dropout => Relu
        Args:
            filters: number of filters
            size: filter size
            norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
            apply_dropout: If True, adds the dropout layer
        Returns:
            Upsample Sequential Model
        """

        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                                   padding='same',
                                                   kernel_initializer=initializer,
                                                   use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result
