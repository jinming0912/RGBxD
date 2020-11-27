import os

os.environ['TF_CUDNN_USE_AUTOTUNE'] = str(0)
import sys
import argparse
from datetime import datetime
from utils import file_util

import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import numpy
import random

random.seed(2)
numpy.random.seed(2)
import tensorflow as tf

tf.random.set_seed(2)

import losses
import settings
import metrics
import callbacks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    args = parser.parse_args()
    setting = settings.load_setting(args.setting)

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(setting.save_folder, '%s_%d_%s' % (args.setting, os.getpid(), time_string))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    setting.model.compile(optimizer=setting.optimizer,
                          loss=losses.SparseCategoricalCrossentropy(class_weight=setting.metrics_weights),
                          metrics=[metrics.LearningRateMetric(setting.optimizer, name='lr'),
                                   metrics.SparseCategoricalAccuracy(class_weight=setting.metrics_weights,
                                                                     name='pixel_acc'),
                                   metrics.SparseCategoricalMeanAccuracy(setting.dataset.num_class,
                                                                         class_weight=setting.metrics_weights,
                                                                         name='mean_acc'),
                                   metrics.SparseCategoricalMeanIoU(setting.dataset.num_class,
                                                                    class_weight=setting.metrics_weights,
                                                                    name='mean_iou')
                                   ])
    # Backup code
    file_util.backup_code(root_folder)
    logging.info("CUDA_VISIBLE_DEVICES: %s" % setting.gpu_id)
    logging.info('PID: %d', os.getpid())
    logging.info(str(args))

    model_callbacks = [
        # Interrupt training if `val_loss` stops improving for over 2 epochs
        # tf.keras.callbacks.EarlyStopping(patience=2, monitor='loss'),
        # Write TensorBoard logs to `root_folder` directory
        tf.keras.callbacks.TensorBoard(log_dir=root_folder),
        # Save the model with best iou
        callbacks.ModelCheckpoint(filepath=os.path.join(root_folder,
                                                        setting.model.name + '-{epoch:03d}-'
                                                                             '{val_mean_iou:.4f}'
                                                                             '.hdf5'),
                                  monitor='val_mean_iou', mode='max',
                                  save_best_only=True, save_weights_only=True)
    ]
    dataset_train = setting.dataset.create_train_dataset()
    dataset_test = setting.dataset.create_test_dataset()
    # custom_entry_flow_conv1_1 = setting.model.get_layer(name='custom_entry_flow_conv1_1')
    # print('custom_entry_flow_conv1_1', custom_entry_flow_conv1_1.get_weights())
    # for layer in setting.model.layers:
    #     layer.trainable = False
    # setting.model.build(input_shape=tuple([2] + setting.dataset.get_input_shape()))
    # setting.model.summary()
    setting.model.fit(dataset_train,
                      # epochs=1,
                      epochs=setting.num_epochs,
                      # steps_per_epoch=2,
                      steps_per_epoch=setting.dataset.get_trainval_size() // setting.batch_size,
                      validation_data=dataset_test,
                      validation_freq=setting.validation_freq,
                      callbacks=model_callbacks
                      )
    # print('custom_entry_flow_conv1_1', custom_entry_flow_conv1_1.get_weights())


if __name__ == '__main__':
    main()
