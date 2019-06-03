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


def write_tensorboard(writer, train_metrics, valid_metrics, args):

    writer.add_scalars('loss', {'train/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                train_metrics['loss'],
                                'valid/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                valid_metrics['loss']},
                       train_metrics['epoch'])
    writer.add_scalars('precision', {'train_50/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                     train_metrics['precision'],
                                     'valid_50/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                     valid_metrics['precision'],
                                     'train_40/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                     train_metrics['precision_40'],
                                     'valid_40/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                     valid_metrics['precision_40'],
                                     'train_60/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                     train_metrics['precision_60'],
                                     'valid_60/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                     valid_metrics['precision_60']},
                       train_metrics['epoch'])

    writer.add_scalars('recall', {'train_50/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                  train_metrics['recall'],
                                  'valid_50/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                  valid_metrics['recall'],
                                  'train_40/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                  train_metrics['recall_40'],
                                  'valid_40/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                  valid_metrics['recall_40'],
                                  'train_60/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                  train_metrics['recall_60'],
                                  'valid_60/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                  valid_metrics['recall_60']},
                       train_metrics['epoch'])

    writer.add_scalars('f1_score', {'train_50/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    train_metrics['f1_score'],
                                    'valid_50/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    valid_metrics['f1_score'],
                                    'train_40/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    train_metrics['f1_score_40'],
                                    'valid_40/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    valid_metrics['f1_score_40'],
                                    'train_60/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    train_metrics['f1_score_60'],
                                    'valid_60/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    valid_metrics['f1_score_60']},
                       train_metrics['epoch'])

    writer.add_scalars('accuracy', {'train_50/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    train_metrics['accuracy'],
                                    'valid_50/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    train_metrics['accuracy'],
                                    'train_40/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    train_metrics['accuracy_40'],
                                    'valid_40/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    train_metrics['accuracy_40'],
                                    'train_60/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    train_metrics['accuracy_60'],
                                    'valid_60/'+args.model+'/'+str(args.batch_normalization)+'/'+str(args.pretrained):
                                    train_metrics['accuracy_60']},
                       train_metrics['epoch'])


def save_weights(model, model_path, ep, train_metrics, valid_metrics):
    torch.save({'model': model.state_dict(),
                'epoch_time': ep,
                'valid_loss': valid_metrics['loss1'],
                'train_loss': train_metrics['loss1']},
               str(model_path)
               )


def write_event(log, train_metrics, valid_metrics):
    CMD='epoch:{} time:{:.2f} train_loss:{:.4f} train_precision:{:.3f} train_recall:{:.3f} train_f1_score:{:.3f} ' \
        'valid_loss:{:.4f} valid_precision:{:.3f} valid_recall: {:.3f} valid_f1_score:{:.3f}'.\
        format(train_metrics['epoch'], train_metrics['epoch_time'], train_metrics['loss'], train_metrics['precision'],
               train_metrics['recall'], train_metrics['f1_score'],
               valid_metrics['loss'], valid_metrics['precision'], valid_metrics['recall'], valid_metrics['f1_score']
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
