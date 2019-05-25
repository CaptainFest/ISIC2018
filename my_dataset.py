import os
import torch
import torchvision.transforms.functional as TF
from keras.preprocessing.image import array_to_img, img_to_array
from torch.utils.data import Dataset, DataLoader
from utils import load_image
import random
import numpy as np
import math


class MyDataset(Dataset):

    def __init__(self, train_test_id, labels_ids, args, train, ids, annotated_np):

        self.train_test_id = train_test_id
        self.image_path = args.image_path
        self.pretrained = args.pretrained
        self.augment_list = args.augment_list
        self.labels_ids = labels_ids
        self.train = train
        self.square_size = args.square_size
        self.mode = args.mode
        self.annotated_np = annotated_np
        if train:
            if ids.size != 0:
                self.labels_ids = self.labels_ids.iloc[ids, :][self.train_test_id['Split'] == 'train'].values.astype('uint8')
                self.train_test_id = self.train_test_id.iloc[ids, :][self.train_test_id['Split'] == 'train'].ID.values
                print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
            else:
                self.labels_ids = self.labels_ids[self.train_test_id['Split'] == 'train'].values.astype('uint8')
                self.train_test_id = self.train_test_id[self.train_test_id['Split'] == 'train'].ID.values
                # print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        else:
            self.labels_ids = self.labels_ids[self.train_test_id['Split'] != 'train'].values.astype('uint8')
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] != 'train'].ID.values
            # print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        self.n = self.train_test_id.shape[0]

    def __len__(self):
        return self.n

    def transform_fn(self, image, mask):

        image = array_to_img(image, data_format="channels_last")
        mask_pil_array = [None] * mask.shape[-1]
        for i in range(mask.shape[-1]):
             mask_pil_array[i] = array_to_img(mask[:, :, i, np.newaxis], data_format="channels_last")

        if 'hflip' in self.augment_list:
            if random.random() > 0.5:
                image = TF.hflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.hflip(mask_pil_array[i])
        if 'vflip' in self.augment_list:
            if random.random() > 0.5:
                image = TF.vflip(image)
                for i in range(mask.shape[-1]):
                    mask_pil_array[i] = TF.vflip(mask_pil_array[i])
        if 'affine' in self.augment_list:
            angle = random.randint(0, 90)
            translate = (random.uniform(0, 100), random.uniform(0, 100))
            scale = random.uniform(0.5, 2)
            image = TF.affine(image, angle, translate, scale)
            for i in range(mask.shape[-1]):
                mask_pil_array[i] = TF.affine(mask_pil_array[i], angle, translate, scale)
        if 'adjust_brightness' in self.augment_list:
            if random.random() > 0.3:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)
        if 'adjust_saturation' in self.augment_list:
            if random.random() > 0.3:
                saturation_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_saturation(image, saturation_factor)

        image = img_to_array(image, data_format="channels_last")
        for i in range(mask.shape[-1]):
            mask[:, :, i] = img_to_array(mask_pil_array[i], data_format="channels_last")[:, :, 0].astype('uint8')

        image = (image / 255.0).astype('float32')
        mask = (mask / 255.0).astype('uint8')

        if self.pretrained:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image - mean)/std

        return image, mask

    def __getitem__(self, index):

        if self.mode == 'grid_AL':
            name = self.train_test_id[np.floor(index / self.square_size**2)]
        else:
            name = self.train_test_id[index]
        path = self.image_path

        # Load image and from h5
        image = load_image(os.path.join(path, '%s.h5' % name), 'image')
        mask = load_image(os.path.join(path, '%s_attribute_all.h5' % name), 'mask')

        if self.train:
            if self.augment_list:
                image, mask = self.transform_fn(image, mask)

        if self.mode == 'grid_AL':
            pass
            #if index:
            #    mask[] = 0
        elif self.mode == 'classic_AL':
            if index in self.annotated_np:
                mask[mask] = 0
        image_with_mask = np.dstack((image, mask))
        labels = self.labels_ids[index, :]

        return image_with_mask, labels, name


class ActiveDataset(Dataset):
    def __init__(self, train_test_id, args, ids):
        self.train_test_id = train_test_id
        self.image_path = args.image_path
        self.train_test_id = self.train_test_id.iloc[ids, :][self.train_test_id['Split'] == 'train'].ID.values
        print('Active_phase ', 'train_test_id.shape: ', self.train_test_id.shape)
        self.n = self.train_test_id.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        name = self.train_test_id[index]
        path = self.image_path
        image = load_image(os.path.join(path, '%s.h5' % (name)), 'image')
        return image, name


def make_loader(train_test_id, labels_ids, args, ids, batch_size, train=True, active_phase=False, shuffle=True,
                annotated_np=np.array([])):

    if not active_phase:
        data_set = MyDataset(train_test_id=train_test_id,
                             labels_ids=labels_ids,
                             args=args,
                             train=train,
                             ids=ids,
                             annotated_np=annotated_np)
    else:
        data_set = ActiveDataset(train_test_id=train_test_id,
                                 args=args,
                                 ids=ids)
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers
                             #pin_memory=torch.cuda.is_available()
                             )
    return data_loader


