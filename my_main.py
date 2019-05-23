import os
import torch.nn
import time
import argparse
# import sklearn.metrics as metrics
from ignite.metrics import Precision, Recall, MetricsLambda

import torch
import torchvision
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
#from utils import save_weights, write_event

from my_dataset import make_loader
from models import create_model
from Active import ActiveLearningTrainer

import pandas as pd
import numpy as np


def f1(r, p):
    return torch.mean(2 * p * r / (p + r + 1e-20)).item()


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='vgg16', choices=['vgg16', 'resnet50', 'resnet152'])
    arg('--batch-normalization', action='store_true')  # if --batch-normalization parameter then True
    arg('--pretrained', action='store_true')           # if --pretrained parameter then True
    arg('--lr', type=int, default=0.001)
    arg('--batch-size', type=int, default=1)
    arg('--workers', type=int, default=1)
    arg('--augment-list', type=list, nargs='*', default=[])
    arg('--image-path', type=str, default='/home/irek/My_work/train/h5_224/')
    arg('--mask-path', type=str, default='/home/irek/My_work/train/binary/')
    arg('--n-epochs', type=int, default=1)
    arg('--K-models', type=int, default=5)
    arg('--begin-number', type=int, default=20)
    arg('--show-model', action='store_true')
    arg('--uncertain_select_num', type=int, default=10)
    arg('--representative_select_num', type=int, default=5)
    arg('--mode', type=str, default='simple', choices=['simple', 'classic_AL', 'grid_AL'])
    args = parser.parse_args()

    epoch = 1
    step = 0

    train_test_id = pd.read_csv('train_test_id.csv')
    mask_ind = pd.read_csv('mask_ind.csv')

    annotated = np.array([])
    non_annotated = np.array([])
    if args.mode in ['classic_AL', 'grid_AL']:
        indexes = train_test_id[train_test_id['Split'] == 'train'].index.tolist()
        annotated = np.random.choice(indexes, args.begin_number, replace=False)
        non_annotated = np.setxor1d(indexes, annotated)

    train_loader = make_loader(train_test_id, mask_ind, args, annotated,  batch_size=args.batch_size, train=True, shuffle=True)
    # valid_loader = make_loader(train_test_id, mask_ind, args, annotated_set,  batch_size=args.batch_size, train=False, shuffle=True)

    if True:
        print('--' * 10)
        print('check data')
        train_image, train_labels_ind, name = next(iter(train_loader))
        print('train_image.shape', train_image.shape)
        print('train_label_ind.shape', train_labels_ind.shape)
        print('train_image.min', train_image.min().item())
        print('train_image.max', train_image.max().item())
        print('train_label_ind.min', train_labels_ind.min().item())
        print('train_label_ind.max', train_labels_ind.max().item())
    print('--' * 10)

    cudnn.benchmark = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    bootstrap_models = {}
    optimizers = {}

    device = ('cuda')

    # define models pool
    for i in range(args.K_models):
        bootstrap_models[i], optimizers[i] = create_model(args, device)

    if args.show_model:
        print(bootstrap_models[0])

    prec = Precision(average=True, is_multilabel=True)
    rec = Recall(average=True, is_multilabel=True)
    prec2 = Precision(is_multilabel=True)
    rec2 = Recall(is_multilabel=True)
    f1_score = MetricsLambda(f1, rec2, prec2)

    criterion = nn.BCEWithLogitsLoss()

    for ep in range(epoch, args.n_epochs + 1):
        try:
            start_time = time.time()
            for model_id in range(args.K_models):
                # state = load_weights(model_id)
                # model.load_state_dict(state['model'])

                ##################################### training #############################################
                if model_id != 0:
                    subset_with_replaces = np.random.choice(annotated, len(annotated), replace=True)
                    print(subset_with_replaces)
                    train_loader = make_loader(train_test_id, mask_ind, args, subset_with_replaces,
                                               batch_size=args.batch_size, train=True, shuffle=True)
                else:
                    train_loader = make_loader(train_test_id, mask_ind, args, annotated,
                                               batch_size=args.batch_size, train=True, shuffle=True)
                n1 = len(train_loader)
                for i, (train_image_batch, train_labels_batch, names) in enumerate(train_loader):
                    if i % 50 == 0:
                        print(f'\rBatch {i} / {n1} ', end='')

                    train_image_batch = train_image_batch.permute(0, 3, 1, 2)
                    train_image_batch = train_image_batch.to(device).type(torch.cuda.FloatTensor)
                    train_labels_batch = train_labels_batch.to(device).type(torch.cuda.FloatTensor)

                    output_probs = bootstrap_models[model_id](train_image_batch)

                    loss = criterion(output_probs, train_labels_batch)

                    optimizers[model_id].zero_grad()  # optimizers[model_id].zero_grad()
                    loss.backward()
                    optimizers[model_id].step()
                    step += 1

                    if model_id == 0:
                        outputs = torch.sigmoid(output_probs)
                        outputs = (outputs > 0.5)

                        prec.update((outputs, train_labels_batch))
                        rec.update((outputs, train_labels_batch))
                        prec2.update((outputs, train_labels_batch))
                        rec2.update((outputs, train_labels_batch))
                # save weights for each model after its training
                # save_weights(model, model_id, args.model_path, epoch + 1, steps)

                epoch_time = time.time() - start_time

                if model_id == 0:
                    train_metrics = {'loss': loss,
                                     'precision': prec.compute(),
                                     'recall': rec.compute(),
                                     'f1_score': f1_score.compute(),
                                     'epoch_time': epoch_time}
                    print('Epoch: {} Loss: {:.6f} Prec: {:.4f} Recall: {:.4f} F1: {:.4f} Time: {:.4f}'.format(
                                                                 ep,
                                                                 train_metrics['loss'],
                                                                 train_metrics['precision'],
                                                                 train_metrics['recall'],
                                                                 train_metrics['f1_score'],
                                                                 train_metrics['epoch_time']))
                    prec.reset()
                    prec2.reset()
                    rec.reset()
                    rec2.reset()
            ##################################### validation ###########################################
            valid_loader = make_loader(train_test_id, mask_ind, args, annotated, batch_size=args.batch_size, train=False, shuffle=True)
            with torch.no_grad():
                n2 = len(valid_loader)

                for i, (valid_image_batch, valid_labels_batch, names) in enumerate(valid_loader):
                    print(f'\rBatch {i} / {n2} ', end='')

                    valid_image_batch = valid_image_batch.permute(0, 3, 1, 2).to(device).type(torch.cuda.FloatTensor)
                    valid_labels_batch = valid_labels_batch.to(device).type(torch.cuda.FloatTensor)

                    output_probs = bootstrap_models[0](valid_image_batch)
                    if ep == args.n_epochs - 1:
                        print(output_probs)

                    outputs = torch.sigmoid(output_probs)
                    outputs = (outputs > 0.5)

                    loss = criterion(output_probs, valid_labels_batch)

                    prec.update((outputs, valid_labels_batch))
                    rec.update((outputs, valid_labels_batch))
                    prec2.update((outputs, valid_labels_batch))
                    rec2.update((outputs, valid_labels_batch))

            valid_metrics = {'loss': loss,
                             'precision': prec.compute(),
                             'recall': rec.compute(),
                             'f1_score': f1_score.compute()}
            print('Epoch: {} Loss: {} Prec: {:.4f} Recall: {:.4f} F1: {:.4f}'.format('\t',
                                                                 valid_metrics['loss'],
                                                                 valid_metrics['precision'],
                                                                 valid_metrics['recall'],
                                                                 valid_metrics['f1_score']))
            print('\n')
            prec.reset()
            prec2.reset()
            rec.reset()
            rec2.reset()

                # write_event(log, epoch=epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)

            if args.mode in ['classic_AL', 'grid_AL']:
                al_trainer = ActiveLearningTrainer(train_test_id, mask_ind, device, args, bootstrap_models, annotated, non_annotated)
                annotated = al_trainer.al_step()
                print(len(non_annotated))
                non_annotated = np.setxor1d(non_annotated, annotated)
                print(len(non_annotated))
                return
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
