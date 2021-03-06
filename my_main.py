import os
import torch.nn
import time
import argparse
import sklearn.metrics as metrics

import torch, torchvision
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam
import torch.nn.functional as F
from utils import save_weights, write_event,print_model_summay,\
    set_freeze_layers,get_freeze_layer_names
import torchvision.models as models

from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, RandomAffine
from my_dataset import make_loader
from loss import LossBinary
from dataset import make_loader
from transforms import ImageOnly, Normalize, HorizontalFlip, VerticalFlip

import load_weights
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', type=str, default='vgg16', choices=['vgg16', 'resnet50'])
    arg('--batch-normalization', type=bool, default=True)
    arg('--pretrained', type=bool, default=False)
    arg('--lr', type=int, default=0.001)
    arg('--batch-size', type=int, default=1)
    arg('--image-path', type=str, default='/home/irek/My_train/h5/')
    arg('--mask-path', type=str, default='/home/irek/My_train/binary/')
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
                    print(model)

            return
            train_test_id = ep

            # define data loader
            train_loader = make_loader(train_test_id, args.image_path, args, train=True, shuffle=True,)
            val_loader   = make_loader(train_test_id, args.image_path, args, train=False, shuffle=True)

            train_image = train_image.permute(0, 3, 1, 2).to(device)
            train_label = train_label.to(device).type(torch.cuda.FloatTensor)

            n_models = 1

            if arg.mode in ['classic_AL', 'grid_AL']:
                n_models = arg.n_models
            for model_id in range(n_models):
                start_time = time.time()
                # state = load_weights(model_id)
                # model.load_state_dict(state['model'])
                for i, (train_image_batch, train_label_batch, train_mask_ind) in enumerate(train_loader):

                    output_probs = model[model_id](train_image)

                    outputs = torch.sigmoid(output_probs)

                    loss = nn.BCEWithLogitsLoss(output_probs, train_label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                    epoch_time = time.time() - start_time
                    train_metrics = {'precision': metrics.average_precision_score(train_label, outputs),
                                     'recall': metrics.recall_score(train_label, outputs, average='samples'),
                                     'F1_score': metrics.f1_score(train_label, outputs),
                                     'epoch_time': epoch_time}

                # valid_metrics = valid_fn(model, criterion, valid_loader, device, num_classes)
                ##############
                ## write events
                # write_event(log, step, epoch=epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)

                # save weights for each model after its training
                # save_weights(model, model_id, args.model_path, epoch + 1, steps)
            if arg.mode in ['classic_AL', 'grid_AL']:
                pass
                # trainer = Trainer(data_path, mask_path)
                # trainer.AL_step()
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    main()
