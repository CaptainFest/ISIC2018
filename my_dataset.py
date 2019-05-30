import os
import torch
import torchvision.transforms.functional as TF
from keras.preprocessing.image import array_to_img, img_to_array
from torch.utils.data import Dataset, DataLoader
from utils import load_image
import random
import numpy as np


class MyDataset(Dataset):

    def __init__(self, train_test_id, labels_ids, args, train, ids):

        self.train_test_id = train_test_id
        self.labels_ids = labels_ids
        self.image_path = args.image_path
        self.pretrained = args.pretrained
        self.augment_list = args.augment_list
        self.train = train
        self.square_size = args.square_size
        self.mode = args.mode
        self.ids = ids
        if train == 'train' or train == 'active':
            if self.ids.size != 0:
                self.labels_ids = self.labels_ids[self.train_test_id['Split'] == 'train'].iloc[self.ids, :].values.astype('uint8')
                self.train_test_id = self.train_test_id[self.train_test_id['Split'] == 'train'].iloc[self.ids, :].ID.values
                print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
            else:
                self.labels_ids = self.labels_ids[self.train_test_id['Split'] == 'train'].values.astype('uint8')
                self.train_test_id = self.train_test_id[self.train_test_id['Split'] == 'train'].ID.values
                print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        elif train == 'valid':
            self.labels_ids = self.labels_ids[self.train_test_id['Split'] != 'train'].values.astype('uint8')
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] != 'train'].ID.values
            print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)

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

        name = self.train_test_id[index]
        path = self.image_path
        # Load image and from h5
        image = load_image(os.path.join(path, '%s.h5' % name), 'image')
        mask = load_image(os.path.join(path, '%s_attribute_all.h5' % name), 'mask')

        if self.train == 'train':
            if self.augment_list:
                image, mask = self.transform_fn(image, mask)

        if self.train == 'active':
            if index in self.ids:
                mask.fill(0.)
        image_with_mask = np.dstack((image, mask))
        labels = self.labels_ids[index, :]

        return image_with_mask, labels, name


class GridDataset(Dataset):

    def __init__(self, train_test_id, labels_ids, args, train, squares_annotated, squares_non_annotated):

        self.train_test_id = train_test_id
        self.labels_ids = labels_ids
        self.image_path = args.image_path
        self.pretrained = args.pretrained
        self.augment_list = args.augment_list
        self.train = train
        self.square_size = args.square_size
        self.mode = args.mode
        self.squares_annotated = squares_annotated
        self.squares_non_annotated = squares_non_annotated
        if train == 'train' or train == 'active':
            self.labels_ids = self.labels_ids[self.train_test_id['Split'] == 'train'].values.astype('uint8')
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] == 'train'].ID.values
            print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)
        elif train == 'valid':
            self.labels_ids = self.labels_ids[self.train_test_id['Split'] != 'train'].values.astype('uint8')
            self.train_test_id = self.train_test_id[self.train_test_id['Split'] != 'train'].ID.values
            print('Train =', self.train, 'train_test_id.shape: ', self.train_test_id.shape)

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

        name = self.train_test_id[index]
        path = self.image_path
        # Load image and from h5
        image = load_image(os.path.join(path, '%s.h5' % name), 'image')
        mask = load_image(os.path.join(path, '%s_attribute_all.h5' % name), 'mask')

        if self.train == 'train':
            if self.augment_list:
                image, mask = self.transform_fn(image, mask)

        if self.train == 'active':
            sqna = self.squares_non_annotated
            indxs = sqna[np.logical_and(sqna > index*(224//self.square_size)**2,
                                        sqna < (index+1)*(224//self.square_size)**2)]
            for i in indxs:
                w = i // self.square_size
                h = i % self.square_size
                mask[w, h] = 0.

        image_with_mask = np.dstack((image, mask))
        labels = self.labels_ids[index, :]

        return image_with_mask, labels, name


def make_loader(train_test_id, labels_ids, args, ids=np.array([]), train=True,
                squares_annotated=np.array([]), squares_non_annotated=np.array([]), shuffle=True):
    if args.mode == 'grid_AL':
        data_set = GridDataset(train_test_id=train_test_id,
                               labels_ids=labels_ids,
                               args=args,
                               train=train,
                               squares_annotated=squares_annotated,
                               squares_non_annotated=squares_non_annotated)
    else:
        data_set = MyDataset(train_test_id=train_test_id,
                             labels_ids=labels_ids,
                             args=args,
                             train=train,
                             ids=ids)
    data_loader = DataLoader(data_set,
                             batch_size=args.batch_size,
                             shuffle=shuffle,
                             num_workers=args.workers,
                             pin_memory=torch.cuda.is_available()
                             )
    return data_loader


