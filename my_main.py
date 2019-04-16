import models, os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn
import my_dataset
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = '/home/ge209/Downloads/ISIC_attribute+18/data'
mask_path = '/home/ge209/Downloads/ISIC_attribute+18/masks'
ds = my_dataset.CustomDataset(data_path, mask_path)
dl = DataLoader(ds, batch_size=2)

model = models.UNet16(num_classes=5, pretrained='vgg').to(device)
loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr = 0.001, momentum=0.9)
n_epochs = 10

for epoch in range(n_epochs):
    print(f'Epoch {epoch} out of {n_epochs}')
    for data in dl:
        inputs, target = data
        inputs = inputs.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)

        optimizer.zero_grad()

        outputs = model(inputs)[0]
        loss_value = loss(outputs, target)

        loss_value.backward()
        optimizer.step()