import pandas as pd
import argparse
import numpy as np
import math
from my_dataset import make_loader


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


def calculate_similarities(train_test_id, mask_ind, args, non_annotated, most_uncertain):
    dl_non_annotated = make_loader(train_test_id, mask_ind, args, non_annotated, batch_size=1, train=True,
                                   active_phase=True, shuffle=False)
    dl_candidates = make_loader(train_test_id, mask_ind, args, most_uncertain, batch_size=1, train=True,
                                active_phase=True, shuffle=False)
    cos_sim_table = np.empty((len(dl_candidates), len(dl_non_annotated)))
    for i, (image2_tensor, _) in enumerate(dl_non_annotated):
        non_annotated_image_np = image2_tensor.cpu().numpy().ravel()
        print(i)
        for k, (image_tensor, _) in enumerate(dl_candidates):
            candidate_image_np = image_tensor.cpu().numpy().ravel()
            cos_sim_table[k, i] = cos_similarity(candidate_image_np, non_annotated_image_np)
    return cos_sim_table


def create_sim_table_file(train_test_id, mask_ind, args, non_annotated, most_uncertain):

    table = calculate_similarities(train_test_id, mask_ind, args, non_annotated, most_uncertain)
    np.save('similarities_table', table)


train_test_id = pd.read_csv('train_test_id.csv')
mask_ind = pd.read_csv('mask_ind.csv')
parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--batch-size', type=int, default=1)
arg('--image-path', type=str, default='/home/irek/My_work/train/h5_112/')
arg('--workers', type=int, default=4)
arg('--mask-path', type=str, default='/home/irek/My_work/train/binary/')
args = parser.parse_args()
non_annotated = train_test_id[train_test_id['Split'] == 'train'].index.tolist()
most_uncertain = non_annotated
create_sim_table_file(train_test_id, mask_ind, args, non_annotated, most_uncertain)
