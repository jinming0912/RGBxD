import logging
import os
import tensorflow as tf

import optimizers
from datasets.nyu_v2_fcn import NyuV2Fcn
from datasets.sun_rgbd import SunRgbd
from models.deeplab_v3_plus import DeeplabV3Plus


class DeeplabV3PlusSetting:
    def __init__(self, setting_path=None):
        self.name = 'DeeplabV3Plus'

        self.gpu_id = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id

        self.enable_eager = True
        if not self.enable_eager:
            tf.compat.v1.disable_eager_execution()

        self.save_folder = os.path.join(os.environ['HOME'], "Summarys")

        self.batch_size = 4
        self.num_epochs = 2000
        self.validation_freq = 2

        self.dataset = NyuV2Fcn(
            batch_size=self.batch_size,
            base_dir=os.path.join(os.environ['HOME'], "Datasets/Downloads/NYU-Depth-V2/nyud"),
            min_scale_factor=0.5,
            max_scale_factor=2,
            scale_factor_step_size=0.25,   # 0.25
            with_aug_flip=True,
            with_aug_scale_crop=True,
            with_aug_gaus_noise=True)
        # self.dataset = SunRgbd(
        #     batch_size=self.batch_size,
        #     base_dir=os.path.join(os.environ['HOME'], "Datasets/sun_rgbd_xyz"),
        #     min_scale_factor=0.5,
        #     max_scale_factor=2,
        #     scale_factor_step_size=0.25,  # 0.25
        #     with_aug_flip=True,
        #     with_aug_scale_crop=True,
        #     with_aug_gaus_noise=True)

        self.loss_weights = [0.0] + [1.0] * (self.dataset.num_class - 1)
        self.metrics_weights = [0.0] + [1.0] * (self.dataset.num_class - 1)

        self.model = DeeplabV3Plus(input_shape=self.dataset.get_input_shape(),
                                   batch_size=self.batch_size,
                                   num_class=self.dataset.num_class,
                                   # weights=os.path.join(os.environ['HOME'], "Models/DeeplabV3Plus-448-0.4875.hdf5"),
                                   weights='cityscapes',  # 'pascal_voc', 'cityscapes', 'imagenet', None
                                   backbone='xception',  # 'xception', 'mobilenetv2'
                                   fusion_mode='bgr_hha_gw',  # bgr, hha, bgr_hha, bgr_hha_xyz, bgr_hha_xyz-gw
                                   with_pre_train_entry_flow_conv1_1=False,
                                   with_pre_train_logits_semantic=False,
                                   batch_normal_trainable=True,
                                   conv_trainable=True,
                                   experiment=None,  # 'gw_weight', None
                                   name=self.name)

        decay_steps = self.num_epochs * (self.dataset.get_trainval_size() // self.batch_size)
        self.lr_schedule = optimizers.schedules.PolynomialDecay(initial_learning_rate=0.003,
                                                                power=0.9,
                                                                decay_steps=decay_steps,
                                                                end_learning_rate=0.0001,
                                                                name='poly_lr')

        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.0015,  # 0.0015, self.lr_schedule
                                                 # decay=0.00001,
                                                 momentum=0.9)
        # self.optimizer = tf.keras.optimizers.Adam(lr=0.002)

        if setting_path is not None:
            # ...
            return
