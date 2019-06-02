import numpy as np
import torch.nn as nn
from my_dataset import make_loader
import torch
import time
from loss import LossBinary
import bottleneck as bn

class ActiveLearningTrainer:

    def __init__(self, train_test_id, mask_ind, device, args, bootstrap_models, annotated=0, non_annotated=0,
                 annotated_squares=0, non_annotated_squares=0):
        self.train_test_id = train_test_id
        self.mask_ind = mask_ind
        self.device = device
        self.args = args
        self.bootstrap_models = bootstrap_models
        if args.mode == 'grid_AL':
            self.annotated_squares = annotated_squares
            self.non_annotated_squares = non_annotated_squares
        else:
            self.annotated = annotated
            self.non_annotated = non_annotated
        self.sims = np.load('similarities_table_64.npy')
        self.uncertain_select_num = args.uncertain_select_num
        self.sq_number = 224 // args.square_size

    def al_classic_step(self):
        if len(self.annotated) == 2294:
            return self.annotated
        if 2294 > len(self.annotated) >= 2294 - self.uncertain_select_num:
            return np.append(self.annotated, self.non_annotated)
        most_uncertain = self.select_uncertain()
        most_representative = self.select_representative(most_uncertain)
        assert(not (set(self.annotated) & set(most_representative)))
        assert(len(self.annotated)+len(most_representative) == len(set(self.annotated) | set(most_representative)))
        return np.append(self.annotated, most_representative)

    def al_grid_step(self):
        annotated_number = len(self.annotated_squares) // self.sq_number**2
        if annotated_number == 2294:
            return self.annotated_squares
        if 2294 > annotated_number >= 2294 - self.uncertain_select_num:
            return np.append(self.annotated_squares, self.non_annotated_squares)
        most_uncertain_squares = self.select_uncertain_square()
        return np.append(self.annotated_squares, most_uncertain_squares)

    def select_uncertain(self):
        criterion = LossBinary(self.args.jaccard_weight)
        train_test_id = self.train_test_id
        mask_ind = self.mask_ind
        args = self.args
        non_annotated = self.non_annotated
        dl = make_loader(train_test_id, mask_ind, args, train='active', ids=non_annotated, shuffle=False)
        most_uncertain_ids = np.zeros([len(non_annotated)])
        for i, (input_, input_labels, names) in enumerate(dl):

            input_tensor = input_.permute(0, 3, 1, 2)
            input_tensor = input_tensor.to(self.device).type(torch.cuda.FloatTensor)
            input_tensor.requires_grad = True
            input_labels = input_labels.to(self.device).type(torch.cuda.FloatTensor)
            grad = torch.zeros([input_tensor.shape[0]], dtype=torch.float64, device=self.device)

            for model_id in range(1, args.K_models):
                out = self.bootstrap_models[model_id](input_tensor)
                self.bootstrap_models[model_id].zero_grad()
                loss = criterion(out, input_labels)
                loss.backward()
                for j in range(input_tensor.shape[0]):
                    grad[j] += input_tensor.grad[j].abs().sum()

            most_uncertain_ids[i * args.batch_size: i * args.batch_size + input_tensor.shape[0]] = grad.cpu()

        indexes = bn.argpartition(-most_uncertain_ids, args.uncertain_select_num)[:args.uncertain_select_num]
        top_uncertain = non_annotated[indexes]

        return top_uncertain

    def select_uncertain_square(self):
        criterion = LossBinary(self.args.jaccard_weight)
        train_test_id = self.train_test_id
        mask_ind = self.mask_ind
        args = self.args
        dl = make_loader(train_test_id, mask_ind, args, train='active', shuffle=False)
        g = len(dl)
        print(g)
        sq_num = 224 // args.square_size
        sq_siz = args.square_size
        squares_to_select = args.uncertain_select_num * sq_num ** 2
        all_uncertainties = np.zeros([2294 * sq_num**2])
        for i, (input_, input_labels, names) in enumerate(dl):
            input_tensor = input_.permute(0, 3, 1, 2)
            input_tensor = input_tensor.to(self.device).type(torch.cuda.FloatTensor)
            input_tensor.requires_grad = True
            input_labels = input_labels.to(self.device).type(torch.cuda.FloatTensor)
            grad = torch.zeros([input_tensor.shape[0] * sq_num**2], dtype=torch.float64, device=self.device)
            for model_id in range(1, args.K_models):
                out = self.bootstrap_models[model_id](input_tensor)
                self.bootstrap_models[model_id].zero_grad()
                loss = criterion(out, input_labels)
                loss.backward()
                for j in range(input_tensor.shape[0]):
                    for w in range(sq_num):
                        for h in range(sq_num):
                            temp = input_tensor.grad[j, :, w * sq_siz:(w + 1) * sq_siz,
                                                        h * sq_siz:(h + 1) * sq_siz].abs().sum()
                            grad[j*sq_num**2 + w*sq_num + h] += temp

            all_uncertainties[i * args.batch_size * sq_num**2:
                               i * args.batch_size * sq_num**2 + input_tensor.shape[0] * sq_num**2] = grad.cpu()

        non_annotated_uncertainties = all_uncertainties[self.non_annotated_squares]
        indexes = bn.argpartition(-non_annotated_uncertainties, squares_to_select)[:squares_to_select]
        top_uncertain = self.non_annotated_squares[indexes]

        return top_uncertain

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
