import logging
import os


class BaseSetting:
    def __init__(self, setting_path=None):
        self.gpu_id = '0'
        self.enable_eager = True
        self.save_folder = os.path.join(os.environ['HOME'], "Summarys")

        self.dataset = DatasetSetting(None)
        self.model = ModelSetting(None)
        self.optimizer = OptimizerSetting(None)

        self.batch_size = 1
        self.num_epochs = 500
        self.loss_weights = [0.0] + [1.0] * (self.dataset.num_class - 1)
        self.metrics_weights = [0.0] + [1.0] * (self.dataset.num_class - 1)

        if setting_path is not None:
            # ...
            return


class DatasetSetting:
    def __init__(self, setting_dict=None):
        self.name = 'nyu_v2_fcn'
        self.dir = os.path.join(os.environ['HOME'], "Datasets/Downloads/NYU-Depth-V2/nyud")
        self.mean_bgr = (122.675, 116.669, 104.008)
        self.mean_hha = (132.431, 94.076, 118.477)
        self.num_class = 41
        self.with_aug_flip = True
        self.with_aug_scale_crop = True
        self.with_aug_gaus_noise = True

        if setting_dict is not None:
            self.name = setting_dict['name']
            self.dir = setting_dict['data_dir']
            self.mean_bgr = setting_dict['mean_bgr']
            self.mean_hha = setting_dict['mean_hha']
            self.mean_logd = setting_dict['mean_logd']
            self.num_class = setting_dict['num_class']
            self.with_data_augmentation = setting_dict['with_data_augmentation']


class OptimizerSetting:
    def __init__(self, setting_dict=None):
        self.name = 'sgd'
        self.learning_rate = 0.00025,
        self.momentum = 0.9

        if setting_dict is not None:
            # ...
            return


class ModelSetting:
    def __init__(self, setting_dict=None):
        self.name = 'DCNN'
        self.mode = 'bgr_hha_one_stream'  # bgr, hha, bgr_hha_one_stream, bgr_hha_two_stream
        self.with_depth_aware = True

        if setting_dict is not None:
            # ...
            return
