import numpy as np
import torch.nn as nn
from my_dataset import make_loader, calculate_similarities
import torch


class ActiveLearningTrainer:

    def __init__(self, train_test_id, mask_ind, device, args, bootstrap_models, annotated, non_annotated):
        self.train_test_id = train_test_id
        self.mask_ind = mask_ind
        self.device = device
        self.args = args
        self.bootstrap_models = bootstrap_models
        self.annotated = annotated
        self.non_annotated = non_annotated

    def al_step(self):
        most_uncertain = self.select_uncertain()
        most_representative = self.select_representative(most_uncertain)
        assert(not (set(self.annotated) & set(most_representative)))
        assert(len(self.annotated)+len(most_representative) == len(set(self.annotated) & set(most_representative)))
        return np.append(self.annotated, most_representative)

    def select_uncertain(self):

        criterion = nn.BCEWithLogitsLoss()
        train_test_id = self.train_test_id
        mask_ind = self.mask_ind
        args = self.args
        non_annotated = self.non_annotated
        dl = make_loader(train_test_id, mask_ind, args, non_annotated, batch_size=1, train=True, shuffle=False)
        most_uncertain_ids = {}
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
                grad = input_tensor.grad.cpu().numpy()
                image_bootstrap_grad += np.sum(abs(grad))
            most_uncertain_ids[non_annotated[i]] = image_bootstrap_grad
        uncertain = sorted(most_uncertain_ids, key=most_uncertain_ids.get, reverse=True)[:args.uncertain_select_num]
        print(uncertain)
        return uncertain

    def select_representative(self, most_uncertain):
        train_test_id = self.train_test_id
        mask_ind = self.mask_ind
        args = self.args
        non_annotated = self.non_annotated
        most_representative = np.array([])
        add_image_id = 0
        with torch.no_grad():
            cos_sim_table = calculate_similarities(train_test_id, mask_ind, args, non_annotated, most_uncertain)
            cand = len(cos_sim_table.shape[0])
            nonannot = len(cos_sim_table.shape[1])
            array_of_maximums = np.zeros(nonannot)
            temp_array = np.zeros(nonannot)
            for i in range(args.representative_select_num):
                max_sum = 0
                print('i=', i)
                for j in range(cand):
                    if most_uncertain[j] not in most_representative:
                        temp_array = array_of_maximums
                        for k in range(nonannot):
                            if cos_sim_table[i, k] > array_of_maximums[k]:
                                temp_array[k] = cos_sim_table[i, k]
                    temp_sum = sum(temp_array)
                    if temp_sum >= max_sum:
                        max_sum = max_sum
                        add_image_id = most_uncertain[j]
                array_of_maximums = temp_array
                most_representative = np.append(most_representative, add_image_id)
        return most_representative
