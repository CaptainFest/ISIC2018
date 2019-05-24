from my_dataset import calculate_similarities
import pandas as pd
import argparse
import numpy as np

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
