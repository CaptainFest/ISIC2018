import models, os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn
import my_dataset
import torch, torchvision
import numpy.random
import typing
from my_trainer import Trainer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = '/home/ge209/Downloads/ISIC_attribute+18/data'
mask_path = '/home/ge209/Downloads/ISIC_attribute+18/masks'




trainer = Trainer(data_path, mask_path)
trainer.AL_step()
torchvision.models.vgg16()