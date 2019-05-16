import os

import torch
import torchvision.transforms.functional as TF
from keras.preprocessing.image import array_to_img, img_to_array
from torch.utils.data import Dataset, DataLoader
from utils import load_image
import pandas as pd
import random


class MyDataset(Dataset):

    def __init__(self, train_test_id, image_path, train, pretrained, augment_list):

        self.train_test_id = train_test_id
        self.image_path = image_path
        self.train = train
        self.pretrained = pretrained
        self.augment_list = augment_list
        self.img_IDs = img_IDs
        self.mask_ind = pd.read_csv('mask_ind.csv')

    def __len__(self):
        return len(self.img_IDs)

    def transform_fn(self, image):

        image_net_mean = (0.485, 0.456, 0.406)
        image_net_std = (0.229, 0.224, 0.225)

        if self.pretrained:
            image = TF.normalize(image, image_net_mean, image_net_std)

        image = array_to_img(image, data_format="channels_last")
        if self.train:
            if 'hflip' in self.augment_list:
                if random.random() > 0.5:
                    image = TF.hflip(image)
            if 'vflip' in self.augment_list:
                if random.random() > 0.5:
                    image = TF.vflip(image)
            if 'affine' in self.augment_list:
                if random.random() > 0.5:
                    angle = random.randint(0, 90)
                    translate = (random.uniform(0, 100), random.uniform(0, 100))
                    scale = random.uniform(0.5, 2)
                    image = TF.affine(image, angle=angle, translate=translate, scale=scale)
            if 'adjust_brightness' in self.augment_list:
                if random.random() > 0.5:
                    brightness_factor = random.uniform(0.8, 1.2)
                    image = TF.adjust_brightness(image, brightness_factor)
            if 'adjust_saturation' in self.augment_list:
                if random.random() > 0.5:
                    saturation_factor = random.uniform(0.8, 1.2)
                    image = TF.adjust_saturation(image, saturation_factor)
        else:
            if self.pretrained:
                image = TF.normalize(image, image_net_mean, image_net_std)

        image = img_to_array(image, data_format="channels_last")
        image = (image / 255.0).astype('float32')

        return image

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.img_IDs[index]
        path = self.image_path

        # Load image from h5
        image = load_image(os.path.join(path, ID, '.h5'))

        if self.train:
            image = self.transform_fn(image)

        label = self.labels[ID]

        return image, label


def make_loader(train_test_id, image_path, args, train=True, shuffle=True):
    data_set = MyDataset(train_test_id=train_test_id,
                         image_path=image_path,
                         train=train,
                         pretrained=args.pretrained,
                         augment_list=args.augment_list)
    data_loader = DataLoader(data_set,
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers,
                             pin_memory=torch.cuda.is_available())
    return data_loader