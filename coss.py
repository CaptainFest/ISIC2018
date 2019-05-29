import pandas as pd
import numpy as np
import math
import argparse
from utils import load_image

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--image_path', type=str, default="/home/irek/My_work/train/h5_64/")
arg('--save_path', type=str, default="/home/irek/My_work/train/h5_64/")
args = parser.parse_args()



def cos_similarity(v1, v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]
        y = v2[i]
        sumxx += x * x
        sumyy += y * y
        sumxy += x * y
    return sumxy / math.sqrt(sumxx * sumyy)


def calculate_similarities(ind, img_id, candidate_list):

    print(f'\r{ind}', end='')
    ###############
    ### load image
    image_file = args.image_path + '%s.h5' % img_id
    image_non_annot = load_image(image_file, 'image').ravel()

    for k, candidate_image_name in enumerate(candidate_list):
        if k <= ind:
            candidate_image_np = load_image(args.image_path + '%s.h5' % candidate_image_name, 'image').ravel()
            cos_sim_table[k, ind] = cos_similarity(candidate_image_np, image_non_annot)
            cos_sim_table[ind, k] = cos_sim_table[k, ind]
    return None


def get_ind(img_name):
    return img_name.split('.')[0]


cos_sim_table = np.ones((2294, 2294), dtype='float')

train_test_id = pd.read_csv('train_test_id.csv')

non_annotated_image_names = train_test_id[train_test_id['Split'] == 'train'].ID.values

img_inds = list(map(get_ind, non_annotated_image_names))

for ind, name in enumerate(img_inds):
    calculate_similarities(ind, name, non_annotated_image_names)

np.save('/data/ISIC/similarities_table', cos_sim_table)
