from torch import nn
import torch
from torchvision import models
import torchvision
from torch.optim import Adam
from torch.nn import functional as F


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-1])
        num_feats = model.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(num_feats, num_classes)
        )

    def forward(self, x):
        f = self.features(x)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


def create_model(args, device):
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

    for param in model.parameters():
        param.requires_grad = False

    if args.model == 'resnet50':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)
    elif args.model == 'vgg16':
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 5)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    return model, optimizer
