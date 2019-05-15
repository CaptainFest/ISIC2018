import os
import torch.nn
import time
import argparse

import torch, torchvision
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam
import torch.nn.functional as F
from utils import save_weights, write_event,print_model_summay,\
    set_freeze_layers,get_freeze_layer_names

import my_dataset
from loss import LossBinary
from my_trainer import Trainer
from dataset import make_loader
from models import UNet16, UNet16BN
from metrics import AllInOneMeter
from transforms import DualCompose,ImageOnly,Normalize,HorizontalFlip,VerticalFlip

import load_weights
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='UNet16BN', choices=['UNet16', 'UNet16BN'])
    arg('--batch-size', type=int, default=1)
    arg('--image-path', type=str, default='/home/irek/My_train/h5/')
    arg('--mask-path', type=str, default='/home/irek/My_train/binary/')
    arg('--n-epochs', type=int, default=1)
    arg('--n-models', type=int, default=5)
    args = parser.parse_args()

    num_classes = 5

    if args.model == 'UNet16':
        model = UNet16(num_classes=num_classes, pretrained='vgg')
    elif args.model == 'UNet16BN':
        model = UNet16BN(num_classes=num_classes, pretrained='vgg')

    ## multiple GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    epoch = 1
    step = 0
    meter = AllInOneMeter()

    loss_fn = LossBinary(jaccard_weight=args.jaccard_weight)

    cudnn.benchmark = True

    train_transform = DualCompose([
        HorizontalFlip(),
        VerticalFlip(),
        ImageOnly(Normalize())
    ])

    val_transform = DualCompose([
        ImageOnly(Normalize())
    ])

    optimizer = Adam(model.parameters(), lr=args.lr)

    criterion = loss_fn

    w1 = 1.0
    w2 = 0.5
    w3 = 0.5

    for ep in range(epoch, args.n_epochs + 1):
        try:
            if epoch == 1:
                freeze_layer_names = get_freeze_layer_names(model, part='encoder')
                set_freeze_layers(model, freeze_layer_names=freeze_layer_names)
                print_model_summay(model)
            elif epoch == 50:
                set_freeze_layers(model, freeze_layer_names=None)
                print_model_summay(model)

            train_test_id = ep

            ## define data loader
            train_loader = make_loader(train_test_id, args.image_path, args, train=True, shuffle=True,
                                       transform=train_transform)
            valid_loader = make_loader(train_test_id, args.image_path, args, train=False, shuffle=True,
                                       transform=val_transform)

            train_image = train_image.permute(0, 3, 1, 2)
            train_mask = train_mask.permute(0, 3, 1, 2)
            train_image = train_image.to(device)
            train_mask = train_mask.to(device).type(torch.cuda.FloatTensor)
            train_mask_ind = train_mask_ind.to(device).type(torch.cuda.FloatTensor)

            for model_id in range(args.n_models):
                start_time = time.time()
                state = load_weights(model_id)
                model.load_state_dict(state['model'])
                for i, (train_image, train_mask, train_mask_ind) in enumerate(train_loader):

                    outputs, outputs_mask_ind1, outputs_mask_ind2 = model[model_id](train_image)

                    train_prob = torch.sigmoid(outputs)
                    train_mask_ind_prob1 = torch.sigmoid(outputs_mask_ind1)
                    train_mask_ind_prob2 = torch.sigmoid(outputs_mask_ind2)

                    loss1 = criterion(outputs, train_mask)
                    loss2 = F.binary_cross_entropy_with_logits(outputs_mask_ind1, train_mask_ind)
                    loss3 = F.binary_cross_entropy_with_logits(outputs_mask_ind2, train_mask_ind)

                    loss = loss1 * w1 + loss2 * w2 + loss3 * w3

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                    meter.add(train_prob, train_mask, train_mask_ind_prob1, train_mask_ind_prob2, train_mask_ind,
                              loss1.item(), loss2.item(), loss3.item(), loss.item())

                epoch_time = time.time() - start_time
                train_metrics = meter.value()
                train_metrics['epoch_time'] = epoch_time
                train_metrics['image'] = train_image.data
                train_metrics['mask'] = train_mask.data
                train_metrics['prob'] = train_prob.data

                #valid_metrics = valid_fn(model, criterion, valid_loader, device, num_classes)
                ##############
                ## write events
                #write_event(log, step, epoch=epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)

                # save weights for each model after its training
                save_weights(model, model_id, args.model_path, epoch + 1, steps)

            #trainer = Trainer(data_path, mask_path)
            #trainer.AL_step()
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
