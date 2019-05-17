import os
import torch.nn
import time
import argparse
import sklearn.metrics as metrics

import torch
import torchvision
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam
import torch.nn.functional as F
from utils import save_weights, write_event,print_model_summay,\
    set_freeze_layers,get_freeze_layer_names
import torchvision.models as models

from my_dataset import make_loader

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='vgg16', choices=['vgg16', 'resnet50'])
    arg('--batch-normalization', type=bool, default=True)
    arg('--pretrained', type=bool, default=False)
    arg('--lr', type=int, default=0.001)
    arg('--batch-size', type=int, default=1)
    arg('--workers', type=int, default=1)
    arg('--augment-list', type=list, nargs='*', default=[])
    arg('--image-path', type=str, default='/home/irek/My_work/train/h5/')
    arg('--mask-path', type=str, default='/home/irek/My_work/train/binary/')
    arg('--n-epochs', type=int, default=1)
    arg('--n-models', type=int, default=5)
    arg('--mode', type=str, default='simple', choices=['simple', 'classic_AL', 'grid_AL'])
    args = parser.parse_args()

    num_classes = 5

    ## load model
    if args.model == 'vgg16':
        if args.pretrained:
            if args.batch_normalization:
                model = models.vgg16_bn(pretrained=True)
            else:
                model = models.vgg16(pretrained=True)
        else:
            if args.batch_normalization:
                model = models.vgg16_bn()
            else:
                model = models.vgg16()
    elif args.model == 'resnet50':
        if args.pretrained:
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet50()
    else:
        return

    # multiple GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    epoch = 1
    step = 0

    train_test_id = pd.read_csv('train_test_id.csv')
    train_loader = make_loader(train_test_id, args.image_path, args, train=True, shuffle=True, )
    valid_loader = make_loader(train_test_id, args.image_path, args, train=False, shuffle=True)

    if True:
        print('--' * 10)
        print('check data')
        train_image, train_mask, train_mask_ind = next(iter(train_loader))
        print('train_image.shape', train_image.shape)
        print('train_mask.shape', train_mask.shape)
        print('train_mask_ind.shape', train_mask_ind.shape)
        print('train_image.min', train_image.min().item())
        print('train_image.max', train_image.max().item())
        print('train_mask.min', train_mask.min().item())
        print('train_mask.max', train_mask.max().item())
        print('train_mask_ind.min', train_mask_ind.min().item())
        print('train_mask_ind.max', train_mask_ind.max().item())
    print('--' * 10)

    cudnn.benchmark = True

    optimizer = Adam(model.parameters(), lr=args.lr)

    for ep in range(epoch, args.n_epochs + 1):
        try:
            # freeze model layers if pretrained
            if args.pretrained:
                for param in model.features.parameters():
                    param.require_grad = False
                if args.model == 'vgg16':
                    num_features = model.classifier[6].in_features
                    features = list(model.classifier.children())[:-1]  # Remove last layer
                    features.extend([nn.Linear(num_features, 5)])  # Add our layer with 4 outputs
                    model.classifier = nn.Sequential(*features)
                    # print(model)

            # return

            """train_test_id = pd.read_csv('train_test_id.csv')

            # define data loader
            train_loader = make_loader(train_test_id, args.image_path, args, train=True, shuffle=True,)
            valid_loader = make_loader(train_test_id, args.image_path, args, train=False, shuffle=True)"""

            n_models = 1

            if args.mode in ['classic_AL', 'grid_AL']:
                n_models = args.n_models
            # define models pool
            models_pool = [model for i in range(n_models)]
            optimizers = [Adam(model.parameters(), lr=args.lr) for i in range(n_models)]

            for model_id in range(n_models):
                start_time = time.time()
                # state = load_weights(model_id)
                # model.load_state_dict(state['model'])
                train_metrics, valid_metrics = 0, 0
                ##################################### training #############################################
                n1 = len(train_loader)
                print(n1)
                for i, (train_image_batch, train_labels_batch) in enumerate(train_loader):
                    print('ay')
                    if i % 20 == 0:
                        print(f'\rBatch {i} / {n1}', end='')

                    train_image_batch = train_image_batch.to(device).type(torch.cuda.FloatTensor).permute(0, 3, 1, 2)
                    train_labels_batch = train_labels_batch.to(device).type(torch.cuda.FloatTensor)

                    output_probs = models_pool[model_id](train_image_batch)

                    outputs = torch.sigmoid(output_probs)

                    loss = nn.BCEWithLogitsLoss(output_probs, train_labels_batch)

                    optimizers[model_id].zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                    epoch_time = time.time() - start_time
                    train_metrics = {'precision': metrics.average_precision_score(train_labels_batch, outputs, average='samples'),
                                     'recall': metrics.recall_score(train_labels_batch, outputs, average='samples'),
                                     'F1_score': metrics.f1_score(train_labels_batch, outputs),
                                     'epoch_time': epoch_time}

                ##################################### validation ###########################################
                with torch.no_grad():
                    n2 = len(valid_loader)
                    for i, (valid_image_batch, valid_labels_batch) in enumerate(valid_loader):
                        print(f'\rBatch {i} / {n2}', end='')
                        valid_image_batch = valid_image_batch.to(device).type(torch.cuda.FloatTensor).permute(0, 3, 1, 2)
                        valid_labels_batch = valid_labels_batch.to(device).type(torch.cuda.FloatTensor)

                        output_probs = models_pool[model_id](valid_image_batch)

                        outputs = torch.sigmoid(output_probs)

                        loss = nn.BCEWithLogitsLoss(output_probs, valid_labels_batch)
                        valid_metrics = {'precision': metrics.average_precision_score(valid_labels_batch, outputs, average='samples'),
                                         'recall': metrics.recall_score(valid_labels_batch, outputs, average='samples'),
                                         'F1_score': metrics.f1_score(valid_labels_batch, outputs)}

                # write events
                # write_event(log, step, epoch=epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)

                # save weights for each model after its training
                # save_weights(model, model_id, args.model_path, epoch + 1, steps)
            if args.mode in ['classic_AL', 'grid_AL']:
                pass
                # trainer = Trainer(data_path, mask_path)
                # trainer.AL_step()
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
