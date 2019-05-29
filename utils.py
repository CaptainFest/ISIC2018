import json

import numpy as np
import h5py
import torch, torchvision
import pandas as pd
import os
import sys


def load_image(file_name, type='image'):
    f = h5py.File(file_name, 'r')
    file_np = f['img'][()]
    if type == 'image':
        file_np = (file_np / 255.0).astype('float32')
    elif type == 'mask':
        file_np = file_np.astype('uint8')
    else:
        print('not choosed type to load')
        return
    return file_np


def write_tensorboard(writer, train_metrics, valid_metrics):

    writer.add_scalars('loss', {'train': train_metrics['loss'], 'valid': valid_metrics['loss']}, train_metrics['epoch'])
    writer.add_scalars('sigm_precision', {'train': train_metrics['sigm_precision'], 'valid': valid_metrics['sigm_precision']},
                       train_metrics['epoch'])
    writer.add_scalars('tanh_precision', {'train': train_metrics['tanh_precision'], 'valid': valid_metrics['tanh_precision']},
                       train_metrics['epoch'])
    writer.add_scalars('sigm_recall', {'train': train_metrics['sigm_recall'], 'valid': valid_metrics['sigm_recall']},
                       train_metrics['epoch'])
    writer.add_scalars('tanh_recall', {'train': train_metrics['tanh_recall'], 'valid': valid_metrics['tanh_recall']},
                       train_metrics['epoch'])
    writer.add_scalars('sigm_f1_score', {'train': train_metrics['sigm_f1_score'], 'valid': valid_metrics['sigm_f1_score']},
                       train_metrics['epoch'])
    writer.add_scalars('tanh_f1_score', {'train': train_metrics['tanh_f1_score'], 'valid': valid_metrics['tanh_f1_score']},
                       train_metrics['epoch'])


def save_weights(model, model_path, ep, train_metrics, valid_metrics):
    torch.save({'model': model.state_dict(), 'epoch_time': ep, 'valid_loss': valid_metrics['loss1'], 'train_loss': train_metrics['loss1']},
               str(model_path)
               )


def write_event(log, train_metrics, valid_metrics):
    CMD='epoch:{} time:{:.2f} train_loss:{:.4f} train_precision:{:.3f} train_recall:{:.3f} train_f1_score:{:.3f} ' \
        'valid_loss:{:.4f} valid_precision:{:.3f} valid_recall: {:.3f} valid_f1_score:{:.3f}'.\
        format(train_metrics['epoch'], train_metrics['epoch_time'], train_metrics['loss'], train_metrics['sigm_precision'],
               train_metrics['sigm_recall'], train_metrics['sigm_f1_score'],
               valid_metrics['loss'], valid_metrics['sigm_precision'], valid_metrics['sigm_recall'], valid_metrics['sigm_f1_score']
    )
    log.write(json.dumps(CMD))
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


class ResultAndArgsSaver:

    def __init__(self, args):
        self.data = pd.DataFrame(columns=['epoch', 'epoch_time', 'loss', 'precision', 'recall', 'f1_score'])
        self.model_name = args.model

    def update(self, results):
        self.data = self.data.append(results)

    def save_train_epoch(self):
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)
        with open('commandline_args.txt', 'w') as f:
            f.write('\n'.join(sys.argv[1:]))
        self.data.to_csv(os.path.join(self.model_name,'train_rez.csv'), index=False)

    def save_valid_epoch(self):
        self.data.to_csv('valid_rez.csv', index=False)
