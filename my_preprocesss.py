import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import h5py
from joblib import Parallel, delayed
import os

mask_path = "/home/irek/My_work/train/binary/"
image_path = "/home/irek/My_work/train/data/"

save_path = "/home/irek/My_work/train/h5_224/"
if not os.path.exists(save_path): os.mkdir(save_path)


def load_image(ind,img_id):
    print(f'\r{ind}', end='')
    ###############
    ### load image
    image_file = image_path + '%s.jpg' % img_id
    img = load_img(image_file, target_size=(224,224), grayscale=False)  # this is a PIL image
    img_np = img_to_array(img)
    ### only 0-255 integers
    img_np = img_np.astype(np.uint8)
    hdf5_file = h5py.File(save_path + '%s.h5' % img_id, 'w')
    hdf5_file.create_dataset('img', data=img_np, dtype=np.uint8)
    hdf5_file.close()
    ################
    return None


img_names = os.listdir(image_path)
img_names = filter(lambda x: x.endswith('jpg'), img_names)

def get_ind(img_name):
    return img_name.split('.')[0]

img_inds = list(map(get_ind, img_names))

results = Parallel(n_jobs=12)(delayed(load_image)(ind,row) for ind,row in enumerate(img_inds))
