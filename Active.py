import numpy as np
import torch.nn as nn
from my_dataset import make_loader
import torch
import time
from loss import LossBinary

class ActiveLearningTrainer:

    def __init__(self, train_test_id, mask_ind, device, args, bootstrap_models, annotated, non_annotated):
        self.train_test_id = train_test_id
        self.mask_ind = mask_ind
        self.device = device
        self.args = args
        self.bootstrap_models = bootstrap_models
        self.annotated = annotated
        self.non_annotated = non_annotated
        self.sims = np.load('similarities_table.npy')
        self.uncertain_select_num = args.uncertain_select_num
        self.sq_number = 224//args.square_size

    def al_classic_step(self):
        if len(self.annotated) == 2294:
            return self.annotated
        if all([len(self.annotated) >= 2294 - self.uncertain_select_num, len(self.annotated) < 2294]):
            return np.append(self.annotated, self.non_annotated)
        most_uncertain = self.select_uncertain()
        most_representative = self.select_representative(most_uncertain)
        assert(not (set(self.annotated) & set(most_representative)))
        assert(len(self.annotated)+len(most_representative) == len(set(self.annotated) | set(most_representative)))
        return np.append(self.annotated, most_representative)

    def al_grid_step(self):
        annotated_number = len(self.annotated) // self.sq_number**2
        if annotated_number == 2294:
            return self.annotated
        if all([annotated_number >= 2294 - self.uncertain_select_num, annotated_number < 2294]):
            return np.append(self.annotated, self.non_annotated)
        most_uncertain = self.select_uncertain()
        return np.append(self.annotated, most_uncertain)

    def select_uncertain(self):
        start = time.time()
        criterion = LossBinary(self.args.jaccard_weight)
        train_test_id = self.train_test_id
        mask_ind = self.mask_ind
        args = self.args
        non_annotated = self.non_annotated
        dl = make_loader(train_test_id, mask_ind, args, train='active', ids=non_annotated, batch_size=args.batch_size, shuffle=False)
        most_uncertain_ids = {}
        for i, (input_, input_labels, names) in enumerate(dl):

            input_tensor = input_.permute(0, 3, 1, 2)
            input_tensor = input_tensor.to(self.device).type(torch.cuda.FloatTensor)
            input_tensor.requires_grad = True
            input_labels = input_labels.to(self.device).type(torch.cuda.FloatTensor)

            grad = torch.zeros([args.batch_size], dtype=torch.float64, device=self.device)
            for model_id in range(1, args.K_models):
                out = self.bootstrap_models[model_id](input_tensor)
                self.bootstrap_models[model_id].zero_grad()
                loss = criterion(out, input_labels)
                loss.backward()
                for j in range(args.batch_size):
                    grad += input_tensor[j].grad.abs().sum()
            for k in range(args.batch_size):
                most_uncertain_ids[non_annotated[i*args.batch_size + k]] = grad[k]
        uncertain = sorted(most_uncertain_ids, key=most_uncertain_ids.get, reverse=True)[:args.uncertain_select_num]
        print(time.time()-start)
        return uncertain

    def select_uncertain_square(self):
        start = time.time()
        criterion = nn.BCEWithLogitsLoss()
        train_test_id = self.train_test_id
        mask_ind = self.mask_ind
        args = self.args
        non_annotated = self.non_annotated
        dl = make_loader(train_test_id, mask_ind, args, train='active', ids=non_annotated, batch_size=1, shuffle=False)
        most_uncertain_ids = {}
        g = len(dl)
        sq = 224 // args.square_size
        w = sq
        h = sq
        grads = torch.zeros([g, w, h], dtype=torch.int32, device=self.device)
        for i, (input_, input_labels, names) in enumerate(dl):
            input_tensor = input_.permute(0, 3, 1, 2)
            input_tensor = input_tensor.to(self.device).type(torch.cuda.FloatTensor)
            input_tensor.requires_grad = True
            input_labels = input_labels.to(self.device).type(torch.cuda.FloatTensor)
            image_bootstrap_grad = 0
            for model_id in range(1, args.K_models):
                out = self.bootstrap_models[model_id](input_tensor)
                self.bootstrap_models[model_id].zero_grad()
                loss = criterion(out, input_labels)
                loss.backward()
                for w in range(224 // args.square_size):
                    for h in range(224 // args.square_size):
                        grads[w, h, i] += input_tensor.grad[w * sq:(w + 1) * sq, h * sq:(h + 1) * sq, :].abs().sum()
                most_uncertain_ids[i * sq ** 2 + w * sq + h] = image_bootstrap_grad
        uncertain = sorted(most_uncertain_ids, key=most_uncertain_ids.get, reverse=True)[:args.uncertain_select_num]
        print(time.time() - start)
        return uncertain

    def select_representative(self, most_uncertain):
        args = self.args
        most_representative = np.array([], dtype='int')
        add_image_id = 0
        with torch.no_grad():
            cos_sim_table = self.sims[most_uncertain, :]
            cand = cos_sim_table.shape[0]
            nonannot = cos_sim_table.shape[1]
            array_of_maximums = np.zeros(nonannot)
            temp_array = np.zeros(nonannot)
            for i in range(args.representative_select_num):
                max_sum = 0
                for j in range(cand):
                    if most_uncertain[j] not in most_representative:
                        temp_array = array_of_maximums
                        for k in range(nonannot):
                            if cos_sim_table[j, k] > array_of_maximums[k]:
                                temp_array[k] = cos_sim_table[j, k]
                        temp_sum = sum(temp_array)
                        if temp_sum >= max_sum:
                            max_sum = temp_sum
                            add_image_id = most_uncertain[j]
                array_of_maximums = temp_array
                most_representative = np.append(most_representative, add_image_id)
        return most_representative
