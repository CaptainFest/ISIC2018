import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import typing

def fill_up(arr):
    h,w,ch = arr.shape
    if h < 512:
        dh_top = (512 - h) // 2
        dh_bottom = 512 - (h - dh_top)
        v_top = arr[0:1,:,:]
        v_bottom = arr[-2:-1,:,:]
        arr = np.concatenate([np.tile(v_top, (dh_top,1,1)),
             arr,
             np.tile(v_bottom, (dh_bottom,1,1))],
            axis=0)
    if w < 512:
        dw_left = (512 - w) // 2
        dw_right = (512 - (w + dw_left))
        v_left = arr[:,0:1,:]
        v_right = arr[:,-2:-1,:]
        arr = np.concatenate([
            np.tile(v_left, (1,dw_left,1)),
            arr,
            np.tile(v_right, (1,dw_right, 1))
        ], axis=1)
    return arr

def center_crop(arr):
    h,w,ch = arr.shape
    dh = (h - 512) // 2
    dw = (w - 512) // 2
    return arr[dh:dh+512, dw:dw+512, :]

def mask_2_channels(z):
    red = (z[:,:,0] == 255) & (z[:,:,1] == 0) & (z[:,:,2] == 0)
    yellow = (z[:,:,0] == 255) & (z[:,:,1] == 255) & (z[:,:,2] == 0)
    green = (z[:,:,0] == 0) & (z[:,:,1] == 255) & (z[:,:,2] == 0)
    cyan = (z[:,:,0] == 0) & (z[:,:,1] == 255) & (z[:,:,2] == 255)
    blue = (z[:,:,0] == 0) & (z[:,:,1] == 0) & (z[:,:,2] == 255)
    return np.stack([red, yellow, green, cyan, blue], axis=-1) * 1

def get_all_pairs(data_path, mask_path):
    def get_idx(img_name): return int(img_name.split('.')[0].split('_')[1])
    def get_files(fp): return map(lambda x: os.path.join(fp, x), sorted(os.listdir(fp), key=get_idx))
    return list(zip(get_files(data_path), get_files(mask_path)))

class CustomDataset(Dataset):
    def __init__(self, all_pairs, indices: typing.List, test_mode = False):
        self.pairs = all_pairs
        self.indices = {i: idx for i, idx in enumerate(sorted(indices))}
        self.test_mode = test_mode

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        true_index = self.indices[idx]
        pair = self.pairs[true_index]
        img = np.rollaxis(center_crop(fill_up(np.array(Image.open(pair[0])))), 2, 0) / 255.
        if self.test_mode:
            return true_index, img
        else:
            mask = np.rollaxis(center_crop(fill_up(mask_2_channels(np.array(Image.open(pair[1]))))), 2, 0)
            return img, mask

