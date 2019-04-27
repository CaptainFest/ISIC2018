import argparse
import json
import random
from pathlib import Path
import numpy as np
import time

## pytorch
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.backends import cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
import torchvision.transforms as transforms

## model
from models import UNet, UNet11, UNet16, UNet16BN, LinkNet34
from loss import LossBinary
from dataset import make_loader
from utils import save_weights, write_valid_event, write_tensorboard,print_model_summay,set_freeze_layers,set_train_layers,get_freeze_layer_names
from validation import validation_binary
from prepare_train_val import get_split
from transforms import DualCompose,ImageOnly,Normalize,HorizontalFlip,VerticalFlip
from metrics import AllInOneMeter

import pandas as pd
import load_weights


def get_model(model_name, num_classes):
    if model_name == 'UNet':
        return UNet(num_classes=num_classes)
    elif model_name == 'UNet11':
        return UNet11(num_classes=num_classes, pretrained='vgg')
    elif model_name == 'UNet16':
        return UNet16(num_classes=num_classes, pretrained='vgg')
    elif model_name == 'UNet16BN':
        return UNet16BN(num_classes=num_classes, pretrained='vgg')
    elif model_name == 'LinkNet34':
        return LinkNet34(num_classes=num_classes, pretrained=True)
    else:
        return UNet(num_classes=num_classes, input_channels=3)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', type=float, default=1)
    arg('--root', type=str, default='runs/debug', help='checkpoint root')
    arg('--image-path', type=str, default='data', help='image path')
    arg('--batch-size', type=int, default=16)
    arg('--workers', type=int, default=10)
    arg('--model', type=str, default='UNet16', choices=['UNet', 'UNet11', 'UNet16', 'UNet16BN', 'LinkNet34'])
    arg('--model-weight', type=str, default=None)
    arg('--attribute', type=str, default='all', choices=['pigment_network', 'negative_network',
                                                              'streaks', 'milia_like_cyst',
                                                              'globules', 'all'])
    args = parser.parse_args()


    ## folder for checkpoint
    root = Path(args.root)
    root.mkdir(exist_ok=True, parents=True)

    image_path = args.image_path

    num_classes = 5 if args.attribute == 'all' else 1
    args.num_classes = num_classes

    ### save initial parameters
    print('--' * 10)
    print(args)
    print('--' * 10)
    root.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    ## load pretrained model
    model = get_model(args.model, num_classes)

    ## load pretrained model
    model_weight = args.model_weight
    state = load_weights.load_weights(model_weight)
    model.load_state_dict(state['model'])
    print('--' * 10)
    print('Load pretrained model', model_weight)
    print('--' * 10)

    ## GPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ## model summary
    print_model_summay(model)

    ## define loss
    loss_fn = LossBinary(jaccard_weight=args.jaccard_weight)


    ## It enables benchmark mode in cudnn.
    ## benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the
    ## optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
    ## But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears,
    ## possibly leading to worse runtime performances.
    cudnn.benchmark = True

    ## get train_test_id
    train_test_id = pd.read_csv('train_test_id.csv')

    ## train vs. val
    print('--' * 10)
    print('num train = {}, num_val = {}'.format((train_test_id['Split'] == 'train').sum(),
                                                (train_test_id['Split'] != 'train').sum()
                                                ))
    print('--' * 10)


    val_transform = DualCompose([
        ImageOnly(Normalize())
    ])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    ## define data loader
    valid_loader = make_loader(train_test_id, image_path, args, train=False, shuffle=True,
                               transform=val_transform)

    valid_fn = validation_binary

    ## loss
    criterion = loss_fn

    #########
    ## start evaluating
    log = root.joinpath('valid.log').open('at', encoding='utf8')
    writer = SummaryWriter()
    meter = AllInOneMeter()
    print('Start training')
    print_model_summay(model)
    meter.reset()
    try:
        valid_metrics = valid_fn(model, criterion, valid_loader, device, num_classes)
        write_valid_event(log, valid_metrics=valid_metrics)

    except KeyboardInterrupt:
        writer.close()
    writer.close()




if __name__ == '__main__':
    main()