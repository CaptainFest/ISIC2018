from torchnet.meter import AUCMeter
import torch
import numpy as np


class AllInOneMeter(object):
    """
    All in one meter: AUC
    """

    def __init__(self):
        #super(AllInOneMeter, self).__init__()
        self.loss1 = []
        self.loss2 = []
        self.loss3 = []
        self.loss = []
        self.jaccard = []
        #self.nbatch = 0
        self.epsilon = 1e-15
        self.intersection = torch.zeros([5], dtype=torch.float, device='cuda:0')
        self.union = torch.zeros([5], dtype=torch.float, device='cuda:0')
        self.reset()

    def reset(self):
        #self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        #self.targets = torch.LongTensor(torch.LongStorage()).numpy()
        self.loss1 = []
        self.loss2 = []
        self.loss3 = []
        self.loss = []
        self.jaccard = []
        self.intersection = torch.zeros([5], dtype=torch.float, device='cuda:0')
        self.union = torch.zeros([5], dtype=torch.float, device='cuda:0')
        #self.nbatch = 0


    def add(self, mask_prob, true_mask, mask_ind_prob1, mask_ind_prob2, true_mask_ind, loss1,loss2,loss3,loss):

        self.loss1.append(loss1)
        self.loss2.append(loss2)
        self.loss3.append(loss3)
        self.loss.append(loss)
        #self.nbatch += true_mask.shape[0]
        y_pred = (mask_prob>0.3).type(true_mask.dtype)
        y_true = true_mask
        self.intersection += (y_pred * y_true).sum(dim=-2).sum(dim=-1).sum(dim=0)
        self.union += y_true.sum(dim=-2).sum(dim=-1).sum(dim=0) + y_pred.sum(dim=-2).sum(dim=-1).sum(dim=0)


    def value(self):
        jaccard_array = (self.intersection / (self.union - self.intersection + self.epsilon))
        #jaccard_array = jaccard_array.data.cpu().numpy()
        jaccard = jaccard_array.mean()
        metrics = {
                   'loss1':np.mean(self.loss1), 'loss2':np.mean(self.loss2),
                   'loss3':np.mean(self.loss3), 'loss':np.mean(self.loss),
                   'jaccard':jaccard.item(), 'jaccard1':jaccard_array[0].item(),'jaccard2':jaccard_array[1].item(),
                   'jaccard3':jaccard_array[2].item(), 'jaccard4':jaccard_array[3].item(),'jaccard5':jaccard_array[4].item(),
                   }
        for mkey in metrics:
            metrics[mkey] = round(metrics[mkey], 4)
        return metrics