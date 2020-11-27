import os
import tensorflow as tf
from datasets.nyu_v2_fcn import NyuV2Fcn
from datasets.sun_rgbd import SunRgbd
from models.u_net import UNet


class UNetSetting:
    def __init__(self, setting_path=None):
        self.name = 'UNet'

        self.gpu_id = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_id

        self.enable_eager = True
        if not self.enable_eager:
            tf.compat.v1.disable_eager_execution()

        self.save_folder = os.path.join(os.environ['HOME'], "Summarys")

        self.batch_size = 3
        self.num_epochs = 2000
        self.validation_freq = 2

        self.dataset = NyuV2Fcn(
            batch_size=self.batch_size,
            img_size=[640, 480],
            base_dir=os.path.join(os.environ['HOME'], "Datasets/Downloads/NYU-Depth-V2/nyud"),
            min_scale_factor=0.5,
            max_scale_factor=2,
            scale_factor_step_size=0.25,
            with_aug_flip=True,
            with_aug_scale_crop=True,
            with_aug_gaus_noise=True)
        # self.dataset = SunRgbd(
        #     batch_size=self.batch_size,
        #     base_dir=os.path.join(os.environ['HOME'], "Datasets/sun_rgbd"),
        #     min_scale_factor=0.5,
        #     max_scale_factor=2,
        #     scale_factor_step_size=0.25,  # 0.25
        #     with_aug_flip=True,
        #     with_aug_scale_crop=True,
        #     with_aug_gaus_noise=True)

        self.loss_weights = [0.0] + [1.0] * (self.dataset.num_class - 1)
        self.metrics_weights = [0.0] + [1.0] * (self.dataset.num_class - 1)

        self.model = UNet(num_class=self.dataset.num_class,
                          fusion_mode='bgr_hha_gw',
                          # 'bgr_hha', 'bgr', 'hha', 'bgr_hha_gw', bgr_hha_xyz
                          name='UNet',
                          backbone='mobile_net_v2',
                          backbone_initializer='imagenet')

        decay_steps = self.num_epochs * self.dataset.get_trainval_size()
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=0.00015,
                                                                         power=0.9,
                                                                         decay_steps=decay_steps,
                                                                         end_learning_rate=0.00001,
                                                                         name='poly_lr')
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001,  # 0.001, self.lr_schedule
                                                 # decay=0.0005,
                                                 momentum=0.9)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        if setting_path is not None:
            # ...
            return
