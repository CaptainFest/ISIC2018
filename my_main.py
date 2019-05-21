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
from utils import save_weights, write_event

from my_dataset import make_loader
from models import create_model

import pandas as pd


def output_tn(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


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
    arg('--n-models', type=int, default=5)
    arg('--show-model', action='store_true')
    arg('--mode', type=str, default='simple', choices=['simple', 'classic_AL', 'grid_AL'])
    args = parser.parse_args()

    epoch = 1
    step = 0

    train_test_id = pd.read_csv('train_test_id.csv')
    train_loader = make_loader(train_test_id, args.image_path, args, train=True, shuffle=True)
    valid_loader = make_loader(train_test_id, args.image_path, args, train=False, shuffle=True)

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
    n_models = args.n_models

    device = ('cuda')

    # define models pool
    for i in range(n_models):
        bootstrap_models[i], optimizers[i] = create_model(args, device)

    if args.show_model:
        print(bootstrap_models[0])

    for ep in range(epoch, args.n_epochs + 1):
        try:
            start_time = time.time()
            for model_id in range(n_models):
                # state = load_weights(model_id)
                # model.load_state_dict(state['model'])

                prec = Precision(average=True, is_multilabel=True)
                rec = Recall(average=True, is_multilabel=True)
                prec2 = Precision(is_multilabel=True)
                rec2 = Recall(is_multilabel=True)
                f1_score = MetricsLambda(f1, rec2, prec2)
                ##################################### training #############################################
                n1 = len(train_loader)
                for i, (train_image_batch, train_labels_batch, names) in enumerate(train_loader):
                    if i % 50 == 0:
                        print(f'\rBatch {i} / {n1} ', end='')

                    train_image_batch = train_image_batch.permute(0, 3, 1, 2)
                    train_image_batch = train_image_batch.to(device).type(torch.cuda.FloatTensor)
                    train_labels_batch = train_labels_batch.to(device).type(torch.cuda.FloatTensor)

                    output_probs = bootstrap_models[model_id](train_image_batch)

                    loss = nn.BCEWithLogitsLoss()
                    loss = loss(output_probs, train_labels_batch)

                    optimizers[model_id].zero_grad()  # optimizers[model_id].zero_grad()
                    loss.backward()
                    optimizers[model_id].step()
                    step += 1
                    #print(train_labels_batch)
                    #print(output_probs)
                    #sf = torch.nn.Softmax(dim=1)
                    #print(sf(output_probs))
                    outputs = torch.sigmoid(output_probs)
                    outputs = (outputs > 0.5)

                    prec.update((outputs, train_labels_batch))
                    rec.update((outputs, train_labels_batch))
                    prec2.update((outputs, train_labels_batch))
                    rec2.update((outputs, train_labels_batch))
                # save weights for each model after its training
                # save_weights(model, model_id, args.model_path, epoch + 1, steps)

                epoch_time = time.time() - start_time

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

                        criterion = nn.BCEWithLogitsLoss()
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

                dl= make_loader(train_test_id, args.image_path, args, train=True, shuffle=True)
                for input_, input_labels, names in dl:
                    input_tensor = input_.permute(0, 3, 1, 2)
                    input_tensor = input_tensor.to(device).type(torch.cuda.FloatTensor)
                    input_tensor.requires_grad = True
                    input_labels = input_labels.to(device).type(torch.cuda.FloatTensor)

                    out = bootstrap_models[0](input_tensor)
                    loss = criterion(out, input_labels)
                    loss.backward()
                    print(names)
                    print(input_tensor.grad)
                    print(input_tensor.shape)
                    print('\n')

                    return
                pass
                # trainer = Trainer(data_path, mask_path
                # )
                # trainer.AL_step()
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
