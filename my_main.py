import torch.nn
import time
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import LossBinary
from utils import write_event, write_tensorboard
from my_dataset import make_loader
from models import create_model
from Active import ActiveLearningTrainer
from metrics import Metrics
#import tensorflow as tf

def train(args, results):

    epoch = 0

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    train_test_id = pd.read_csv('train_test_id.csv')
    mask_ind = pd.read_csv('mask_ind.csv')

    annotated = np.array([])
    non_annotated = np.array([])
    annotated_squares = np.array([])
    non_annotated_squares = np.array([])
    K_models = 1
    if args.mode in ['classic_AL', 'grid_AL']:
        len_train = len(train_test_id[train_test_id['Split'] == 'train'])
        indexes = np.arange(len_train)
        np.random.seed(42)
        annotated = np.sort(np.random.choice(indexes, args.begin_number, replace=False))
        temp = [111, 208, 402, 408, 422, 430, 602, 759, 782, 913,
                1288, 1290, 1349, 1726, 1731, 1825, 1847, 2060, 2160, 2285]
        assert(set(annotated).intersection(temp))
        if args.mode == 'grid_AL':
            square_size = (224 // args.square_size)**2
            squares_indexes = np.arange(len_train * square_size)
            annotated_squares = np.array([np.arange(an*square_size, (an+1)*square_size)
                                          for an in annotated]).ravel()
            non_annotated_squares = np.array(list(set(squares_indexes) - set(annotated_squares)))
        non_annotated = np.array(list(set(indexes) - set(annotated)))
        K_models = args.K_models
    train_loader = make_loader(train_test_id, mask_ind, args, annotated,  train='train', shuffle=True)

    if False:
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

    device = 'cuda'
    if args.cuda1:
        device = 'cuda:1'

    # define models pool
    for i in range(K_models):
        bootstrap_models[i], optimizers[i] = create_model(args, device)

    if args.show_model:
        print(bootstrap_models[0])

    criterion = LossBinary(args.jaccard_weight)

    log = root.joinpath('train.log').open('at', encoding='utf8')
    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    scheduler = ReduceLROnPlateau(optimizers[0], 'min', factor=0.8, patience=10, verbose=True)

    writer = SummaryWriter()
    metric = Metrics()

    for ep in range(epoch, args.n_epochs):
        try:
            start_time = time.time()
            for model_id in range(K_models):
                if args.pretrained:
                    if ep == 50:
                        for param in bootstrap_models[model_id].parameters():
                            param.requires_grad = True
                ##################################### training #############################################
                if model_id != 0:
                    subset_with_replaces = np.random.choice(annotated, len(annotated), replace=True)
                    train_loader = make_loader(train_test_id, mask_ind, args, ids=subset_with_replaces, train='train',
                                               shuffle=True)
                else:
                    train_loader = make_loader(train_test_id, mask_ind, args, ids=annotated, train='train',
                                               shuffle=True)
                n1 = len(train_loader)

                for i, (train_image_batch, train_labels_batch, names) in enumerate(train_loader):
                    if i % 50 == 0:
                        print(f'\rBatch {i} / {n1} ', end='')
                    elif i >= n1 - 50:
                        print(f'\r', end='')

                    train_image_batch = train_image_batch.permute(0, 3, 1, 2)
                    train_image_batch = train_image_batch.to(device).type(torch.cuda.FloatTensor)
                    train_labels_batch = train_labels_batch.to(device).type(torch.cuda.FloatTensor)

                    output_probs = bootstrap_models[model_id](train_image_batch)

                    if isinstance(args.attribute, str) and (args.attribute != 'attribute_all'):
                        train_labels_batch = torch.reshape(train_labels_batch, (-1, 1))

                    loss = criterion(output_probs, train_labels_batch)

                    optimizers[model_id].zero_grad()
                    loss.backward()
                    optimizers[model_id].step()

                    if model_id == 0:
                        outputs = torch.sigmoid(output_probs)
                        metric.update(outputs.detach().cpu().numpy(), train_labels_batch.cpu().numpy())

            epoch_time = time.time() - start_time

            train_metrics = metric.compute_train(loss, ep, epoch_time)
            print('Epoch: {} Loss: {:.6f} Prec: {:.4f} Recall: {:.4f} Time: {:.4f}'.format(
                                                         ep,
                                                         train_metrics['loss'],
                                                         train_metrics['precision'],
                                                         train_metrics['recall'],
                                                         train_metrics['epoch_time']))
            results = results.append({'freeze_mode': args.freezing,
                                      'lr': args.lr,
                                      'exp': args.N,
                                      'train_mode': 'train',
                                      'epoch': ep,
                                      'loss': train_metrics['loss'],
                                      'prec': train_metrics['precision'],
                                      'recall': train_metrics['recall']}, ignore_index=True)

            metric.reset()
            ##################################### validation ###########################################
            valid_loader = make_loader(train_test_id, mask_ind, args, train='valid', shuffle=True)
            with torch.no_grad():
                n2 = len(valid_loader)
                for i, (valid_image_batch, valid_labels_batch, names) in enumerate(valid_loader):
                    if i == n2-1:
                        print(f'\r', end='')
                    elif i < n2-3:
                        print(f'\rBatch {i} / {n2} ', end='')
                    valid_image_batch = valid_image_batch.permute(0, 3, 1, 2).to(device).type(torch.cuda.FloatTensor)
                    valid_labels_batch = valid_labels_batch.to(device).type(torch.cuda.FloatTensor)

                    output_probs = bootstrap_models[0](valid_image_batch)

                    if isinstance(args.attribute, str) and (args.attribute != 'attribute_all'):
                        valid_labels_batch = torch.reshape(valid_labels_batch, (-1, 1))
                    loss = criterion(output_probs, valid_labels_batch)

                    outputs = torch.sigmoid(output_probs)
                    metric.update(outputs.detach().cpu().numpy(), valid_labels_batch.cpu().numpy())

            valid_metrics = metric.compute_valid(loss)
            print('\t\t Loss: {:.6f} Prec: {:.4f} Recall: {:.4f}'.format(
                                                                 valid_metrics['loss'],
                                                                 valid_metrics['precision'],
                                                                 valid_metrics['recall']))

            results = results.append({'freeze_mode': args.freezing,
                                      'lr': args.lr,
                                      'exp': args.N,
                                      'train_mode': 'valid',
                                      'epoch': ep,
                                      'loss': valid_metrics['loss'],
                                      'prec': valid_metrics['precision'],
                                      'recall': valid_metrics['recall']}, ignore_index=True)
            metric.reset()
            write_event(log, train_metrics=train_metrics, valid_metrics=valid_metrics)
            write_tensorboard(writer, train_metrics, valid_metrics, args)
            scheduler.step(valid_metrics['loss'])

            if args.mode in ['classic_AL', 'grid_AL']:
                if args.mode == 'classic_AL':
                    temp_time = time.time()
                    al_trainer = ActiveLearningTrainer(train_test_id, mask_ind, device, args, bootstrap_models,
                                                       annotated=annotated, non_annotated=non_annotated)
                    annotated = al_trainer.al_classic_step()
                    print(len(annotated))
                    non_annotated = np.array(list(set(non_annotated) - set(annotated)))
                    print('classic_time', time.time() - temp_time)
                else:
                    temp_time = time.time()
                    al_trainer = ActiveLearningTrainer(train_test_id, mask_ind, device, args, bootstrap_models,
                                                       annotated_squares=annotated_squares,
                                                       non_annotated_squares=non_annotated_squares)
                    annotated_squares = al_trainer.al_grid_step()
                    print(len(annotated_squares) // (224//args.square_size)**2)
                    non_annotated_squares = np.array(list(set(non_annotated_squares) - set(annotated_squares)))
                    print('grid_time', time.time() - temp_time)
        except KeyboardInterrupt:
            return
    writer.close()
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='vgg16', choices=['vgg16', 'resnet50', 'resnet152', 'inception_v3'])
    arg('--mask_use', action='store_true')
    arg('--root', type=str, default='runs/debug')
    arg('--N', type=int, default=1)
    arg('--batch-normalization', action='store_true')  # if --batch-normalization parameter then True
    arg('--pretrained', action='store_true')           # if --pretrained parameter then True
    arg('--lr', type=float, nargs='*', default=[0.001])
    arg('--batch-size', type=int, default=1)
    arg('--augment-list', type=list, nargs='*', default=[])
    arg('--image-path', type=str, default='/home/irek/My_work/train/h5_224/')
    arg('--n-epochs', type=int, default=1)
    arg('--K-models', type=int, default=5)
    arg('--begin-number', type=int, default=20)
    arg('--show-model', action='store_true')
    arg('--uncertain_select-num', type=int, default=10)
    arg('--representative-select-num', type=int, default=5)
    arg('--square-size', type=int, default=16)
    arg('--jaccard-weight', type=float, default=0.)
    arg('--attribute', type=str, nargs='*', default='attribute_all')
    arg('--mode', type=str, default='simple', choices=['simple', 'classic_AL', 'grid_AL'])
    arg('--freezing', action='store_true')
    arg('--al-pool-train', action='store_true')
    arg('--jac_train', action='store_true')
    arg('--grid_train', action='store_true')
    arg('--cuda1', action='store_true')
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)
    log = root.joinpath('train.log').open('at', encoding='utf8')

    results = pd.DataFrame(columns=['freeze_mode', 'lr', 'exp', 'train_mode', 'epoch', 'loss', 'prec',
                                    'recall'])
    N = args.N
    learning_rates = args.lr
    freeze_modes = [True, False]
    mask_use = [False, True]

    for m_use in mask_use:
        args.mask_use = m_use
        for mode in freeze_modes:
            args.freezing = mode
            for lr in learning_rates:
                args.lr = lr
                for experiment in range(N):
                    args.N = experiment
                    print('Использование масок на трейне {} Заморозка {}, шаг обучения {}, '
                          'номер эксперимента {}'.format(args.mask_use, args.freezing, args.lr, args.N))
                    if args.mode == 'simple':
                        i = 0
                        root.joinpath('params'+str(i)+'.json').write_text(
                            json.dumps(vars(args), indent=True, sort_keys=True))
                        results = train(args, results)
                        i += 1
                    elif args.al_pool_train:
                        configs = {'mode': ['classic_AL', 'grid_AL']}
                        i = 0
                        for m in configs['mode']:
                            args.mode = m
                            root.joinpath('params'+str(i)+'.json').write_text(
                                json.dumps(vars(args), indent=True, sort_keys=True))
                            results = train(args, results)
                            i += 1
                    elif args.jac_train:
                        configs = {'jaccard-weight': [0., 0.5, 1.]}
                        i = 0
                        for m in configs['jaccard-weight']:
                            args.jaccard_weight = m
                            root.joinpath('params' + str(i) + '.json').write_text(
                                json.dumps(vars(args), indent=True, sort_keys=True))
                            results = train(args, results)
                            i += 1
                    elif args.grid_train:
                        configs = {'square_size': [32, 8, 16]}
                        i = 0
                        for s in configs['square_size']:
                            args.square_size = s
                            root.joinpath('params' + str(i) + '.json').write_text(
                                json.dumps(vars(args), indent=True, sort_keys=True))
                            results = train(args, results)
                            i += 1
                    else:
                        print('strange')
    results.to_csv('resnet_all_res', index=False)
