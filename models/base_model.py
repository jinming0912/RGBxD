from encoders.resnet_v1_beta import ResNet50Beta
import tensorflow as tf


class BaseModel(tf.keras.Model):
    def __init__(self, backbone, batch_input_shape, fusion_mode='bgr_hha', dilated=True, multi_grid=False,
                 backbone_weights='imagenet', multi_dilation=None, conv_trainable=True,
                 name='BaseModel'):
        super(BaseModel, self).__init__(name=name)
        # copying modules from pretrained models
        if backbone == 'resnet50':
            self.backbone_model = ResNet50Beta(batch_input_shape,
                                               include_top=False,
                                               weights=backbone_weights,
                                               fusion_mode=fusion_mode,
                                               dilated=dilated,
                                               multi_grid=multi_grid,
                                               multi_dilation=multi_dilation,
                                               conv_trainable=conv_trainable)
            self.backbone_model.build(input_shape=tuple(batch_input_shape))
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def call(self, x, **kwargs):
        # tf.print('layers', self.pretrained.layers)
        # tmp = self.pretrained.layers[2].weights[0]
        # print('layers[2].weights[0]', tmp)
        # tf.print('layers[2].weights[0]', tmp[0])
        # tmp = self.pretrained.layers[3].weights[1]
        # print('layers[3].weights[1]', tmp)
        # tf.print('layers[3].weights[1]', tmp)
        # tmp = self.pretrained.layers[3].weights[2]
        # print('layers[3].weights[2]', tmp)
        # tf.print('layers[3].weights[2]', tmp)
        #
        # tmp = self.pretrained.layers[19].weights[0]
        # print('layers[19].weights[0]', tmp)
        # tf.print('layers[19].weights[0]', tmp[0])
        # tmp = self.pretrained.layers[20].weights[1]
        # print('layers[20].weights[1]', tmp)
        # tf.print('layers[20].weights[1]', tmp)
        #
        # tmp = self.pretrained.layers[-3].weights[1]
        # print('layers[-3].weights[1]', tmp)
        # tf.print('layers[-3].weights[1]', tmp)
        # tmp = self.pretrained.layers[-3].weights[2]
        # print('layers[-3].weights[2]', tmp)
        # tf.print('layers[-3].weights[2]', tmp)
        x = self.backbone_model(x, **kwargs)
        return x
