import os
import numpy as np
from data_preprocess.nyu_v2_fcn import data_utils


def main():
    dir_nyud = "user_home/Disk/datasets/Downloads/NYU-Depth-V2-FCN/nyud"

    train_ids = data_utils.get_train_ids(dir_nyud)
    lines = []
    for id in train_ids:
        line = "{}/data/images/img_{}.png {}/data/label/img_{}.png {}/data/depth/img_{}.png {}/data/hha/img_{}.png"\
            .format(dir_nyud, id, dir_nyud, id, dir_nyud, id, dir_nyud, id)
        lines.append(line)
        path_train = os.path.join(dir_nyud, 'train.lst')
        np.savetxt(path_train, np.array(lines), '%s')

    test_ids = data_utils.get_test_ids(dir_nyud)
    lines = []
    for id in test_ids:
        line = "{}/data/images/img_{}.png {}/data/label/img_{}.png {}/data/depth/img_{}.png {}/data/hha/img_{}.png"\
            .format(dir_nyud, id, dir_nyud, id, dir_nyud, id, dir_nyud, id)
        lines.append(line)
        path_test = os.path.join(dir_nyud, 'test.lst')
        np.savetxt(path_test, np.array(lines), '%s')


if __name__ == '__main__':
    main()