import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys
import argparse
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import tensorflow as tf
from datetime import datetime

import settings

from utils import vis_utils, data_utils_bak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', '-x', help='Setting to use', required=True)
    parser.add_argument('--weights', '-w', help='Weights path', required=True)
    parser.add_argument('--save', '-s', help='save dir', required=False)
    args = parser.parse_args()

    setting = settings.load_setting(args.setting)
    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(setting.save_folder, '%s_%d_%s' % (args.setting, os.getpid(), time_string))
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    dataset = setting.dataset.create_test_dataset(one_batch=True)

    logging.info("CUDA_VISIBLE_DEVICES: %s" % setting.gpu_id)
    logging.info('PID: %d', os.getpid())
    logging.info(str(args))
    model = setting.model
    model.load_weights(filepath=args.weights, by_name=True)

    for id, data in enumerate(dataset):
        x, y = data
        y = np.squeeze(y)
        preds = model.predict(x)
        preds = np.squeeze(np.argmax(preds, axis=-1))
        print(y.shape, preds.shape)
        preds[y == 0] = 0
        preds = vis_utils.visualize_seg(np.array(preds), vis_utils.get_color_map(setting.dataset.num_class),
                                        setting.dataset.num_class) * 255

        labels = vis_utils.visualize_seg(np.array(y), vis_utils.get_color_map(setting.dataset.num_class),
                                         setting.dataset.num_class) * 255

        bgrs = x[:, :, :, 0:3]
        blues = np.expand_dims(bgrs[..., 0] + setting.dataset.mean_rgb[2], axis=-1)
        greens = np.expand_dims(bgrs[..., 1] + setting.dataset.mean_rgb[1], axis=-1)
        reds = np.expand_dims(bgrs[..., 2] + setting.dataset.mean_rgb[0], axis=-1)
        rgbs = np.concatenate((reds, greens, blues), axis=-1)

        hhas = x[:, :, :, 3:6]

        hds = np.expand_dims(hhas[..., 0] + setting.dataset.mean_hha[0], axis=-1)
        hgs = np.expand_dims(hhas[..., 1] + setting.dataset.mean_hha[1], axis=-1)
        angles = np.expand_dims(hhas[..., 2] + setting.dataset.mean_hha[2], axis=-1)
        hhas = np.concatenate((hds, hgs, angles), axis=-1)

        hds = np.tile(hds, (1, 1, 1, 3))
        hgs = np.tile(hgs, (1, 1, 1, 3))
        angles = np.tile(angles, (1, 1, 1, 3))

        deps = np.array(x[..., 6]) * 120
        deps = deps - deps.min()
        deps_ori = deps
        deps = deps / deps.max() * 511
        deps = np.expand_dims(deps, axis=-1)
        deps_r = np.clip(deps - 255, 0, 255)
        deps_g = np.clip(deps, 0, 255) + (255 - np.clip(deps - 255, 0, 255))
        deps_b = 255 - np.clip(deps, 0, 255)
        deps = np.concatenate([deps_r, deps_g, deps_b], axis=-1)

        lines = np.ones(shape=(rgbs.shape[0], rgbs.shape[1], 5, rgbs.shape[3])) * 255
        imgs = np.concatenate((rgbs, lines, hhas, lines, deps, lines, labels, lines, preds), axis=2).astype(np.uint8)

        print('id', id)
        if args.save is None:
            plt.imshow(imgs[0])
            plt.axis('off')  # 不显示坐标轴
            plt.show()
            break
        else:
            img = Image.fromarray(imgs[0])
            if not os.path.exists(args.save):
                os.makedirs(args.save)
            path = os.path.join(args.save, str(id) + '.png')
            img.save(path)


if __name__ == '__main__':
    main()
