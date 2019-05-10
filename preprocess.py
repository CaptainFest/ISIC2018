import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import h5py
from joblib import Parallel, delayed
import os

mask_path = "/home/irek/My_work/train/binary/"
image_path = "/home/irek/My_work/train/data/"

save_path = "/home/irek/My_work/train/h5/"
if not os.path.exists(save_path): os.mkdir(save_path)


def load_image(ind,img_id):
    print(f'\r{ind}', end='')

    ###############
    ### load image
    image_file = image_path + '%s.jpg' % img_id
    img = load_img(image_file, target_size=(512,512), grayscale=False)  # this is a PIL image
    img_np = img_to_array(img)
    ### why only 0-255 integers
    img_np = img_np.astype(np.uint8)
    hdf5_file = h5py.File(save_path + '%s.h5' % img_id, 'w')
    hdf5_file.create_dataset('img', data=img_np, dtype=np.uint8)
    hdf5_file.close()
    ################
    ### load masks
    attr_types = ['pigment_network', 'negative_network', 'streaks', 'milia_like_cyst', 'globules']
    masks = np.zeros(shape=(img_np.shape[0], img_np.shape[1], 5))
    for i, attr in enumerate(attr_types):
        mask_file = mask_path + '%s_attribute_%s.png' % (img_id, attr)
        m = load_img(mask_file, target_size=(512,512), grayscale=True)  # this is a PIL image
        m_np = img_to_array(m)
        masks[:, :, i] = m_np[:, :, 0]
        #m_np = m_np[:, :, 0, np.newaxis]
        #m_np = (m_np / 255).astype('int8')
        #hdf5_file = h5py.File(save_path + '%s_attribute_%s.h5' % (img_id, attr), 'w')
        #hdf5_file.create_dataset('img', data=m_np, dtype=np.int8)
        #hdf5_file.close()
    masks = (masks / 255).astype('int8')
    hdf5_file = h5py.File(save_path + '%s_attribute_all.h5' % (img_id), 'w')
    hdf5_file.create_dataset('img', data=masks, dtype=np.int8)
    hdf5_file.close()
    # print(img_np.shape,masks.shape)
    ##########################
    #masks = masks.astype('uint8')
    return None


img_names = os.listdir(image_path)
img_names = filter(lambda x: x.endswith('jpg'), img_names)

def get_ind(img_name):
    return img_name.split('.')[0]

img_inds = list(map(get_ind, img_names))

results = Parallel(n_jobs=12)(delayed(load_image)(ind,row) for ind,row in enumerate(img_inds))
