import models, os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.nn
import my_dataset
import torch, torchvision
import numpy.random
import typing

def f_set(unannotated_indices, candidate_indices, vec):
    return sum(map(lambda idx: f(idx, candidate_indices, vec), unannotated_indices))


def f(idx, candidate_indices, vec):
    return max(map(lambda i: sim(vec[idx], vec[i]), candidate_indices))


def sim(u, v):
    return np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))


class Trainer:
    def __init__(self, data_path, mask_path, device):
        self.pairs = my_dataset.get_all_pairs(data_path, mask_path)[:200]
        self.n_epochs = 1 # todo this parameter should be changed
        self.annotated = numpy.random.choice(list(range(len(self.pairs))), 20, replace=False)
        self.K = 4 # todo this parameter should be changed
        self.device = device

    def AL_step(self):
        main_model, bootstrap_models = self.train_models()
        most_uncertain = self.select_uncertain(20, bootstrap_models, list(set(range(len(self.pairs))).difference(set(self.annotated))))
        most_representative = self.select_representative(10, main_model, most_uncertain)
        self.annotated += most_representative

    def select_representative(self, num_to_select, model, most_uncertain_indices: typing.List):
        annotated_indices = self.annotated
        unannotated_indices = list(set(range(len(self.pairs))).difference(set(annotated_indices)).difference(set(most_uncertain_indices)))

        vec = {}
        test_ds = my_dataset.CustomDataset(self.pairs, unannotated_indices, test_mode=True)
        dl = DataLoader(test_ds, batch_size=1, shuffle=False)
        with torch.no_grad():
            for data in dl:
                true_index, image = data
                true_index = true_index.cpu().data.numpy()[0]
                image = image.to(self.device, dtype=torch.float)
                vec[true_index] = model(image)[1].cpu().data.numpy()
        selected_indices = []
        for _ in range(num_to_select):
            improvement = {idx: f_set(unannotated_indices, selected_indices.append(idx), vec) for idx in most_uncertain_indices}
            best = max(improvement.items(), key=lambda x: x[1])[0]
            selected_indices.append(best)
        return selected_indices

    def select_uncertain(self, num_to_select, bootstrap_models: typing.Dict, indices: typing.List):
        test_ds = my_dataset.CustomDataset(self.pairs, indices, test_mode=True)
        dl = DataLoader(test_ds, batch_size=1, shuffle=False)
        uncertainty_level = {}
        with torch.no_grad():
            for data in dl:
                true_idx, img = data
                true_idx = true_idx.cpu().data.numpy()[0]
                img = img.to(self.device, dtype=torch.float)
                masks = {}
                for k in bootstrap_models:
                    masks[k] = bootstrap_models[k](img)[0].cpu().data.numpy()
                masks = np.moveaxis(np.concatenate(list(masks.values()),axis=0), (0,1,2,3), (-1,0,1,2))
                std = np.std(masks, axis=-1)
                uncertainty_level[true_idx] = np.mean(std)
        sorted_uncertainty = list(map(lambda x: x[0], sorted(uncertainty_level.items(), key=lambda x: x[1])))
        return sorted_uncertainty[:num_to_select]

    def train_models(self):
        d = 0
        # train model on all labelled for representativeness
        main_model = self.train_model(self.annotated)
        # bootstrap new samples and train models for uncertainty
        bootstrap_models = {}
        for k in range(self.K):
            indices = numpy.random.choice(self.annotated, len(self.annotated), replace=True)
            bootstrap_models[k] = self.train_model(indices)

        return main_model, bootstrap_models



    def train_model(self, labelled_indices):
        ds = my_dataset.CustomDataset(self.pairs, labelled_indices)
        dl = DataLoader(ds, batch_size=2)

        model = models.UNet16(num_classes=5, pretrained='vgg').to(self.device)
        for param in model.conv1.parameters(): param.requires_grad = False
        for param in model.conv2.parameters(): param.requires_grad = False
        for param in model.conv3.parameters(): param.requires_grad = False
        for param in model.conv4.parameters(): param.requires_grad = False
        for param in model.conv5.parameters(): param.requires_grad = False
        optimizer = torch.optim.SGD(params=model.parameters(), lr=0.001, momentum=0.9)
        loss = torch.nn.BCEWithLogitsLoss()


        for epoch in range(self.n_epochs):
            print(f'Epoch {epoch} out of {self.n_epochs}')
            for data in dl:
                inputs, target = data
                inputs = inputs.to(self.device, dtype=torch.float)
                target = target.to(self.device, dtype=torch.float)

                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()

                    outputs = model(inputs)[0]
                    loss_value = loss(outputs, target)

                    loss_value.backward()
                    optimizer.step()
        return model