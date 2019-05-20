import os
import torch.nn
import time
import argparse
# import sklearn.metrics as metrics
from ignite.metrics import Precision, Recall

import torch
import torchvision
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
from utils import save_weights, write_event,print_model_summay

from my_dataset import make_loader
from models import create_model

import pandas as pd

device = 0

def output_tn(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y



def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='vgg16', choices=['vgg16', 'resnet50'])
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
        train_image, train_mask_ind = next(iter(train_loader))
        print('train_image.shape', train_image.shape)
        print('train_label_ind.shape', train_mask_ind.shape)
        print('train_image.min', train_image.min().item())
        print('train_image.max', train_image.max().item())
        print('train_label_ind.min', train_mask_ind.min().item())
        print('train_label_ind.max', train_mask_ind.max().item())
    print('--' * 10)

    cudnn.benchmark = True
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)

    bootstrap_models = {}
    optimizers = {}
    n_models = args.n_models

    device = ('cpu')

    # define models pool
    for i in range(n_models):
        bootstrap_models[i], optimizers[i] = create_model(args, device)

    print(bootstrap_models[0])

    for ep in range(epoch, args.n_epochs + 1):
        try:
            start_time = time.time()
            for model_id in range(n_models):
                # state = load_weights(model_id)
                # model.load_state_dict(state['model'])

                prec = Precision(average=True, is_multilabel=True)
                rec = Recall(average=True, is_multilabel=True)
                ##################################### training #############################################
                n1 = len(train_loader)
                for i, (train_image_batch, train_labels_batch) in enumerate(train_loader):

                    if i % 50 == 0:
                        print(f'\rBatch {i} / {n1}', end='')

                    train_image_batch = train_image_batch.permute(0, 3, 1, 2)
                    train_image_batch = train_image_batch.to(device).type(torch.FloatTensor)
                    train_labels_batch = train_labels_batch.to(device).type(torch.FloatTensor)

                    output_probs = bootstrap_models[model_id](train_image_batch)

                    loss = nn.BCEWithLogitsLoss()
                    loss = loss(output_probs, train_labels_batch)

                    optimizers[model_id].zero_grad()  # optimizers[model_id].zero_grad()
                    loss.backward()
                    optimizers[model_id].step()
                    step += 1
                    # print(output_probs)
                    outputs = (output_probs > 0.5)
                    # print(outputs)
                    prec.update((outputs, train_labels_batch))
                    rec.update((outputs, train_labels_batch))
                # save weights for each model after its training
                # save_weights(model, model_id, args.model_path, epoch + 1, steps)

                epoch_time = time.time() - start_time

                train_metrics = {'precision': prec.compute(),
                                 'recall': rec.compute(),
                                 'epoch_time': epoch_time}
                print(train_metrics)
                prec.reset()
                rec.reset()
                ##################################### validation ###########################################
                with torch.no_grad():
                    n2 = len(valid_loader)
                    prec = Precision(is_multilabel=True)
                    rec = Recall(is_multilabel=True)
                    for i, (valid_image_batch, valid_labels_batch) in enumerate(valid_loader):
                        print(f'\rBatch {i} / {n2}', end='')
                        valid_image_batch = valid_image_batch.permute(0, 3, 1, 2).to(device).type(torch.FloatTensor)
                        valid_labels_batch = valid_labels_batch.to(device).type(torch.FloatTensor)

                        output_probs = bootstrap_models[0](valid_image_batch)

                        outputs = (output_probs > 0.5)

                        criterion = nn.BCEWithLogitsLoss()
                        loss = criterion(output_probs, valid_labels_batch)

                        prec.update((outputs, valid_labels_batch))
                        rec.update((outputs, valid_labels_batch))

                valid_metrics = {'precision':prec.compute(),
                                 'recall': rec.compute()}
                print(valid_metrics)
                prec.reset()
                rec.reset()

                # write_event(log, step, epoch=epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)

            if args.mode in ['classic_AL', 'grid_AL']:

                dl= make_loader(train_test_id, args.image_path, args, train=True, shuffle=True)
                for input_, input_labels in dl:
                    input_tensor = input_.permute(0, 3, 1, 2)
                    input_tensor = input_tensor.to(device).type(torch.FloatTensor)
                    input_tensor.requires_grad = True
                    input_labels = input_labels.to(device).type(torch.FloatTensor)

                    out = bootstrap_models[0](input_tensor)
                    loss = criterion(out, input_labels)
                    gradient = loss.backward()
                    print(gradient)
                    gr2 = input_tensor.backward()
                    print(gr2)
                    return
                return
                pass
                # trainer = Trainer(data_path, mask_path)
                # trainer.AL_step()
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
