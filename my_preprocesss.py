import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import h5py
from joblib import Parallel, delayed
import os

image_path = "/home/irek/My_work/train/data/"
mask_path = "/home/irek/My_work/train/binary/"

save_path = "/home/irek/My_work/train/h5_112/"

if not os.path.exists(save_path):
    os.mkdir(save_path)


def load_image(ind, img_id):
    print(f'\r{ind}', end='')
    ###############
    print(img_id)
    ### load image
    image_file = image_path + '%s.jpg' % img_id
    img = load_img(image_file, target_size=(112,112), color_mode='rgb')  # this is a PIL image
    img_np = img_to_array(img)
    ### only 0-255 integers
    img_np = img_np.astype(np.uint8)
    hdf5_file = h5py.File(save_path + '%s.h5' % img_id, 'w')
    hdf5_file.create_dataset('img', data=img_np, dtype=np.uint8)
    hdf5_file.close()
    ################

    attr_types = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']
    #masks = np.zeros(shape=(img_np.shape[0], img_np.shape[1], 5))
    #for i, attr in enumerate(attr_types):
    #    mask_file = mask_path + '%s_attribute_%s.png' % (img_id, attr)
    #    m = load_img(mask_file, target_size=(299, 299), color_mode="grayscale")  # this is a PIL image
    #    m_np = img_to_array(m)
    #    masks[:, :, i] = m_np[:, :, 0]

    #masks = (masks / 255).astype('int8')
    #masks[masks == 0] = -1
    #hdf5_file = h5py.File(save_path + '%s_attribute_all.h5' % (img_id), 'w')
    #hdf5_file.create_dataset('img', data=masks, dtype=np.int8)
    #hdf5_file.close()
    return None


img_names = os.listdir(image_path)
img_names = filter(lambda x: x.endswith('jpg'), img_names)

def get_ind(img_name):
    return img_name.split('.')[0]

img_inds = list(map(get_ind, img_names))

results = Parallel(n_jobs=8)(delayed(load_image)(ind, row) for ind,row in enumerate(img_inds))
