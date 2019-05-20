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
from torch.optim import Adam
import torch.nn.functional as F
from utils import save_weights, write_event,print_model_summay,\
    set_freeze_layers,get_freeze_layer_names
import torchvision.models as models

from my_dataset import make_loader

import pandas as pd


def output_tn(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


def create_model(args):
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
    return model


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
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device('cpu')
    print(device)
    model.to(device)

    epoch = 1
    step = 0

    train_test_id = pd.read_csv('train_test_id.csv')
    train_loader = make_loader(train_test_id, args.image_path, args, train=True, shuffle=True, )
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

    optimizer = Adam(model.parameters(), lr=args.lr)

    for ep in range(epoch, args.n_epochs + 1):
        try:
            # freeze layers if pretrained
            if args.pretrained:
                for param in model.features.parameters():
                    param.require_grad = False
            # replace last layer
            if args.model == 'vgg16':
                in_f = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(in_f, 5)
                # print(model)
            if args.model == 'resnet50':
                in_f = model.fc.in_features
                model.fc = nn.Linear(in_f, 5)
                print(model)


            n_models = 1
            bootstrap_models = {}

            if args.mode in ['classic_AL', 'grid_AL']:
                n_models = args.n_models
                # define models pool
                for i in range(n_models):
                    bootstrap_models[i] = create_model(args.model)
            else:
                bootstrap_models[0] = create_model(args.model)
            # define models pool

            if ep == 1:
                print(bootstrap_models[0])
            optimizers = Adam(model.parameters(), lr=args.lr) # [Adam(model.parameters(), lr=args.lr) for i in range(n_models)]

            for model_id in range(n_models):
                start_time = time.time()
                # state = load_weights(model_id)
                # model.load_state_dict(state['model'])
                train_metrics, valid_metrics = 0, 0
                prec = Precision()
                rec = Recall()
                ##################################### training #############################################
                n1 = len(train_loader)
                for i, (train_image_batch, train_labels_batch) in enumerate(train_loader):

                    if i % 50 == 0:
                        print(f'\rBatch {i} / {n1}', end='')

                    train_image_batch = train_image_batch.permute(0, 3, 1, 2)
                    train_image_batch = train_image_batch.to(device).type(torch.FloatTensor)
                    train_labels_batch = train_labels_batch.to(device).type(torch.FloatTensor)

                    output_probs = bootstrap_models[](train_image_batch)  # models_pool[model_id](train_image_batch)

                    loss = nn.BCEWithLogitsLoss()
                    loss = loss(output_probs, train_labels_batch)
                    #nn.BCEWithLogitsLoss(output_probs, train_labels_batch)

                    optimizers.zero_grad() # optimizers[model_id].zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1
                    # print(output_probs)
                    outputs = (output_probs > 0.5)
                    # print(outputs)
                    prec.update((outputs, train_labels_batch))
                    rec.update((outputs, train_labels_batch))

                    epoch_time = time.time() - start_time

                train_metrics = {'precision': prec.compute().item(),
                                 'recall': rec.compute().item()}
                print(train_metrics)
                prec.reset()
                rec.reset()
                ##################################### validation ###########################################
                with torch.no_grad():
                    n2 = len(valid_loader)
                    prec = Precision()
                    rec = Recall()
                    for i, (valid_image_batch, valid_labels_batch) in enumerate(valid_loader):
                        print(f'\rBatch {i} / {n2}', end='')
                        valid_image_batch = valid_image_batch.permute(0, 3, 1, 2).to(device).type(torch.FloatTensor)
                        valid_labels_batch = valid_labels_batch.to(device).type(torch.FloatTensor)

                        output_probs = bootstrap_models[0](valid_image_batch)  # models_pool[model_id](valid_image_batch)

                        outputs = (output_probs > 0.5)

                        loss = nn.BCEWithLogitsLoss()
                        loss = loss(output_probs, valid_labels_batch)

                        prec.update((outputs, valid_labels_batch))
                        rec.update((outputs, valid_labels_batch))


                valid_metrics = {'precision': prec.compute().item(),
                                 'recall': rec.compute().item()}
                print(valid_metrics)
                prec.reset()
                rec.reset()

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
