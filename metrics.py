from ignite.metrics import Precision, Recall, Accuracy, MetricsLambda
import torch

def f1(r, p):
    return torch.mean(2 * p * r / (p + r + 1e-20)).item()


class Metrics():

    def __init__(self):

        self.prec = Precision(average=True, is_multilabel=True)
        self.prec_40 = Precision(average=True, is_multilabel=True)
        self.prec_60 = Precision(average=True, is_multilabel=True)

        self.rec = Recall(average=True, is_multilabel=True)
        self.rec_40 = Recall(average=True, is_multilabel=True)
        self.rec_60 = Recall(average=True, is_multilabel=True)

        self.prec_f1 = Precision(is_multilabel=True)
        self.rec_f1 = Recall(is_multilabel=True)
        self.f1_score = MetricsLambda(f1, self.rec_f1, self.prec_f1)

        self.prec_f1_40 = Precision(is_multilabel=True)
        self.rec_f1_40 = Recall(is_multilabel=True)
        self.f1_score_40 = MetricsLambda(f1, self.rec_f1_40, self.prec_f1_40)

        self.prec_f1_60 = Precision(is_multilabel=True)
        self.rec_f1_60 = Recall(is_multilabel=True)
        self.f1_score_60 = MetricsLambda(f1, self.rec_f1_60, self.prec_f1_60)

    def update(self,outputs, labels_batch):

        outputs1 = (outputs > 0.5)
        self.prec.update((outputs1, labels_batch))
        self.rec.update((outputs1, labels_batch))
        self.prec_f1.update((outputs1, labels_batch))
        self.rec_f1.update((outputs1, labels_batch))

        outputs_40 = (outputs > 0.4)
        self.prec_40.update((outputs_40, labels_batch))
        self.rec_40.update((outputs_40, labels_batch))
        self.prec_f1_40.update((outputs_40, labels_batch))
        self.rec_f1_40.update((outputs_40, labels_batch))

        outputs_60 = (outputs > 0.6)
        self.prec_60.update((outputs_60, labels_batch))
        self.rec_60.update((outputs_60, labels_batch))
        self.prec_f1_60.update((outputs_60, labels_batch))
        self.rec_f1_60.update((outputs_60, labels_batch))

    def reset(self):

        self.prec.reset()
        self.prec_40.reset()
        self.prec_60.reset()

        self.rec.reset()
        self.rec_40.reset()
        self.rec_60.reset()

        self.prec_f1.reset()
        self.prec_f1_40.reset()
        self.prec_f1_60.reset()
        self.rec_f1.reset()
        self.rec_f1_40.reset()
        self.rec_f1_60.reset()

