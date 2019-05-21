import json
import re
from pathlib import Path

import random
import numpy as np
import h5py
import torch
import torchvision


def load_image(image_file):
    f = h5py.File(image_file, 'r')
    img_np = f['img'][()]
    img_np = (img_np / 255.0).astype('float32')
    return img_np


def load_mask(image_path, img_id, attribute='pigment_network'):
    if attribute == 'all':
        mask_file = image_path + '%s_attribute_all.h5' % (img_id)
        f = h5py.File(mask_file, 'r')
        mask_np = f['img'].value
    else:
        mask_file = image_path + '%s_attribute_%s.h5' % (img_id, attribute)
        f = h5py.File(mask_file, 'r')
        mask_np = f['img'].value

    mask_np = mask_np.astype('uint8')
    return mask_np


def save_weights(model, model_id, model_path, ep, step, train_metrics, valid_metrics):
    torch.save({'model': model.state_dict(), 'model_id':model_id, 'epoch': ep, 'step': step, 'valid_loss': valid_metrics['loss1'], 'train_loss': train_metrics['loss1']},
               str(model_path)
               )

def write_event(log, step, epoch, train_metrics, valid_metrics):
    CMD = 'epoch:{} step:{} time:{:.2f} \n train_loss:{:.3f} {:.3f} {:.3f} {:.3f} train_auc1:{} {} {} {} {} train_auc2:{} {} {} {} {} train_jaccard:{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} \n valid_loss:{:.3f} {:.3f} {:.3f} {:.3f} valid_auc1:{} {} {} {} {} valid_auc2:{} {} {} {} {} valid_jaccard:{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
        epoch, step, train_metrics['epoch_time'],
        train_metrics['loss'],train_metrics['loss1'],train_metrics['loss2'],train_metrics['loss3'],
        train_metrics['jaccard'],train_metrics['jaccard1'],train_metrics['jaccard2'],train_metrics['jaccard3'],train_metrics['jaccard4'],train_metrics['jaccard5'],
        valid_metrics['loss'], valid_metrics['loss1'], valid_metrics['loss2'], valid_metrics['loss3'],
        valid_metrics['jaccard'],valid_metrics['jaccard1'],valid_metrics['jaccard2'],valid_metrics['jaccard3'],valid_metrics['jaccard4'],valid_metrics['jaccard5'],
    )
    print(CMD)
    log.write(json.dumps(CMD))
    # log.write(json.dumps(valid_metrics, sort_keys=True))
    log.write('\n')
    log.flush()


def write_valid_event(log, valid_metrics):
    CMD = 'valid_loss:{:.3f} {:.3f} {:.3f} {:.3f} valid_auc1:{} {} {} {} {} valid_auc2:{} {} {} {} {} valid_jaccard:{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
        valid_metrics['loss'], valid_metrics['loss1'], valid_metrics['loss2'], valid_metrics['loss3'],
        valid_metrics['jaccard'],valid_metrics['jaccard1'],valid_metrics['jaccard2'],valid_metrics['jaccard3'],valid_metrics['jaccard4'],valid_metrics['jaccard5'],
    )
    print(CMD)
    log.write(json.dumps(CMD))
    # log.write(json.dumps(valid_metrics, sort_keys=True))
    log.write('\n')
    log.flush()


def set_freeze_layers(model, freeze_layer_names=None):
    for name, para in model.named_parameters():
        if freeze_layer_names is None:
            para.requires_grad = True
        else:
            if name in freeze_layer_names:
                print('Freeze layer->', name)
                para.requires_grad = False
            else:
                para.requires_grad = True


def set_train_layers(model, train_layer_names=None):
    for name, para in model.named_parameters():
        if train_layer_names is None:
            para.requires_grad = False
        else:
            if name in train_layer_names:
                print('Train layer ->', name)
                para.requires_grad = True
            else:
                para.requires_grad = False


def get_freeze_layer_names(model,part):
    freeze_layers = []
    for ind, (name, para) in enumerate(model.named_parameters()):
        if re.search(part, name):
            #print(ind, name, para.numel(), para.requires_grad)
            freeze_layers.append(name)
        if part == 'encoder':
            if re.search('center', name):
                freeze_layers.append(name)
    return freeze_layers