import os
from PIL import Image
from data_preprocess.nyu_v2_fcn import data_utils



def main():
    dir_nyud = "user_home/Datasets/Downloads/NYU-Depth-V2/nyud"
    dir_label_img = "user_home/Datasets/Downloads/NYU-Depth-V2/nyud/data/label"
    if not os.path.exists(dir_label_img):
        os.makedirs(dir_label_img)

    label_ids = data_utils.get_all_ids(dir_nyud)
    for id in label_ids:
        label_mat = data_utils.load_label(dir_nyud, id)
        img = Image.fromarray(label_mat)
        path_out = '{}/img_{}.png'.format(dir_label_img, id)
        img.save(path_out)




if __name__ == '__main__':
    main()
