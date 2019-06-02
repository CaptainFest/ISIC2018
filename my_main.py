import torch.nn
import time
import argparse
from pathlib import Path
from ignite.metrics import Precision, Recall, MetricsLambda
import pandas as pd
import numpy as np
import json

import torch
from torch.backends import cudnn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss import LossBinary
from utils import write_event, write_tensorboard
from my_dataset import make_loader
from models import create_model
from Active import ActiveLearningTrainer


def f1(r, p):
    return torch.mean(2 * p * r / (p + r + 1e-20)).item()


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='vgg16', choices=['vgg16', 'resnet50', 'resnet152', 'inception_v3'])
    arg('--root', type=str, default='runs/debug')
    arg('--batch-normalization', action='store_true')  # if --batch-normalization parameter then True
    arg('--pretrained', action='store_true')           # if --pretrained parameter then True
    arg('--lr', type=float, default=0.001)
    arg('--batch-size', type=int, default=1)
    arg('--workers', type=int, default=1)
    arg('--augment-list', type=list, nargs='*', default=[])
    arg('--image-path', type=str, default='/home/irek/My_work/train/h5_224/')
    arg('--mask-path', type=str, default='/home/irek/My_work/train/binary/')
    arg('--n-epochs', type=int, default=1)
    arg('--K-models', type=int, default=5)
    arg('--begin-number', type=int, default=20)
    arg('--show-model', action='store_true')
    arg('--uncertain_select-num', type=int, default=10)
    arg('--representative-select-num', type=int, default=5)
    arg('--square-size', type=int, default=16)
    arg('--jaccard-weight', type=float, default=0.)
    arg('--conv-learn-enabled', action='store_true')   # if --conv-learn-enabled parameter then True
    arg('--mode', type=str, default='simple', choices=['simple', 'classic_AL', 'grid_AL'])
    args = parser.parse_args()

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
        annotated = sorted(np.random.choice(indexes, args.begin_number, replace=False))
        if args.mode == 'grid_AL':
            square_size = (224 // args.square_size)**2
            squares_indexes = np.arange(len_train * square_size)
            annotated_squares = sorted(np.random.choice(indexes, args.begin_number, replace=False))
            annotated_squares = np.array([np.arange(an*square_size, (an+1)*square_size)
                                          for an in annotated_squares]).ravel()
            non_annotated_squares = np.array(list(set(squares_indexes) - set(annotated_squares)))

        non_annotated = np.array(list(set(indexes) - set(annotated)))
        K_models = args.K_models

    assert(set(non_annotated).intersection(sorted(non_annotated)))

    train_loader = make_loader(train_test_id, mask_ind, args, annotated,  train='train', shuffle=True)

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
    for i in range(K_models):
        bootstrap_models[i], optimizers[i] = create_model(args, device)

    if args.show_model:
        print(bootstrap_models[0])

    sigm_prec = Precision(average=True, is_multilabel=True)
    sigm_rec = Recall(average=True, is_multilabel=True)
    sigm_prec2 = Precision(is_multilabel=True)
    sigm_rec2 = Recall(is_multilabel=True)
    sigm_f1_score = MetricsLambda(f1, sigm_rec2, sigm_prec2)
    sigm_40_prec = Precision(average=True, is_multilabel=True)
    sigm_60_prec = Precision(average=True, is_multilabel=True)

    criterion = LossBinary(args.jaccard_weight)

    log = root.joinpath('train.log').open('at', encoding='utf8')
    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))
    scheduler = ReduceLROnPlateau(optimizers[0], 'min', factor=0.8, patience=10, verbose=True)

    writer = SummaryWriter()

    for ep in range(epoch, args.n_epochs):
        try:
            start_time = time.time()
            for model_id in range(K_models):
                # state = load_weights(model_id)
                # model.load_state_dict(state['model'])
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

                    loss = criterion(output_probs, train_labels_batch)

                    optimizers[model_id].zero_grad()
                    loss.backward()
                    optimizers[model_id].step()

                    if model_id == 0:

                        outputs = torch.sigmoid(output_probs)
                        outputs1 = (outputs > 0.5)
                        sigm_prec.update((outputs1, train_labels_batch))
                        sigm_rec.update((outputs1, train_labels_batch))
                        sigm_prec2.update((outputs1, train_labels_batch))
                        sigm_rec2.update((outputs1, train_labels_batch))
                        outputs2 = (outputs > 0.4)
                        sigm_40_prec.update((outputs2, train_labels_batch))
                        outputs3 = (outputs > 0.6)
                        sigm_60_prec.update((outputs3, train_labels_batch))

                # save weights for each model after its training
                # save_weights(model, model_id, args.model_path, epoch + 1, steps)

                epoch_time = time.time() - start_time

            train_metrics = {'epoch': ep,
                             'loss': loss,
                             'sigm_precision': sigm_prec.compute(),
                             'sigm_recall': sigm_rec.compute(),
                             'sigm_f1_score': sigm_f1_score.compute(),
                             'sigm_40_precision': sigm_40_prec.compute(),
                             'sigm_60_precision': sigm_60_prec.compute(),
                             'epoch_time': epoch_time}
            print('Epoch: {} Loss: {:.6f} Prec: {:.4f} Recall: {:.4f} F1: {:.4f} Time: {:.4f}'.format(
                                                         train_metrics['epoch'],
                                                         train_metrics['loss'],
                                                         train_metrics['sigm_precision'],
                                                         train_metrics['sigm_recall'],
                                                         train_metrics['sigm_f1_score'],
                                                         train_metrics['epoch_time']))
            sigm_prec.reset()
            sigm_prec2.reset()
            sigm_rec.reset()
            sigm_rec2.reset()
            sigm_40_prec.reset()
            sigm_60_prec.reset()

            ##################################### validation ###########################################
            valid_loader = make_loader(train_test_id, mask_ind, args, train='valid', shuffle=True)
            with torch.no_grad():
                n2 = len(valid_loader)

                for i, (valid_image_batch, valid_labels_batch, names) in enumerate(valid_loader):
                    if i == n2-2:
                        print(f'\r', end='')
                    elif i < n2-2:
                        print(f'\rBatch {i} / {n2} ', end='')
                    valid_image_batch = valid_image_batch.permute(0, 3, 1, 2).to(device).type(torch.cuda.FloatTensor)
                    valid_labels_batch = valid_labels_batch.to(device).type(torch.cuda.FloatTensor)

                    output_probs = bootstrap_models[0](valid_image_batch)

                    loss = criterion(output_probs, valid_labels_batch)

                    outputs = torch.sigmoid(output_probs)
                    outputs1 = (outputs > 0.5)
                    sigm_prec.update((outputs1, valid_labels_batch))
                    sigm_rec.update((outputs1, valid_labels_batch))
                    sigm_prec2.update((outputs1, valid_labels_batch))
                    sigm_rec2.update((outputs1, valid_labels_batch))
                    outputs2 = (outputs > 0.4)
                    sigm_40_prec.update((outputs2, valid_labels_batch))
                    outputs3 = (outputs > 0.6)
                    sigm_60_prec.update((outputs3, valid_labels_batch))

            valid_metrics = {'loss': loss,
                             'sigm_precision': sigm_prec.compute(),
                             'sigm_recall': sigm_rec.compute(),
                             'sigm_f1_score': sigm_f1_score.compute(),
                             'sigm_40_precision': sigm_40_prec.compute(),
                             'sigm_60_precision': sigm_60_prec.compute()
                             }
            print('\t\t Loss: {:.6f} Prec: {:.4f} Recall: {:.4f} F1: {:.4f}'.format(
                                                                 valid_metrics['loss'],
                                                                 valid_metrics['sigm_precision'],
                                                                 valid_metrics['sigm_recall'],
                                                                 valid_metrics['sigm_f1_score']))
            sigm_prec.reset()
            sigm_prec2.reset()
            sigm_rec.reset()
            sigm_rec2.reset()
            sigm_40_prec.reset()
            sigm_60_prec.reset()

            write_event(log, train_metrics=train_metrics, valid_metrics=valid_metrics)

            write_tensorboard(writer, train_metrics, valid_metrics)

            scheduler.step(valid_metrics['loss'])

            if args.mode in ['classic_AL', 'grid_AL']:
                if args.mode == 'classic_AL':
                    temp_time = time.time()
                    al_trainer = ActiveLearningTrainer(train_test_id, mask_ind, device, args, bootstrap_models,
                                                       annotated=annotated, non_annotated=non_annotated)
                    annotated = al_trainer.al_classic_step()
                    non_annotated = np.array(list(set(non_annotated) - set(annotated)))
                    print(time.time() - temp_time)
                else:
                    temp_time = time.time()
                    al_trainer = ActiveLearningTrainer(train_test_id, mask_ind, device, args, bootstrap_models,
                                                       annotated_squares=annotated_squares,
                                                       non_annotated_squares=non_annotated_squares)
                    annotated_squares = al_trainer.al_grid_step()
                    non_annotated_squares = np.array(list(set(non_annotated_squares) - set(annotated_squares)))
                    print(time.time() - temp_time)
        except KeyboardInterrupt:
            return
    writer.close()


if __name__ == "__main__":
    main()
