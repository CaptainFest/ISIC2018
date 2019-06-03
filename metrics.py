from ignite.metrics import Precision, Recall, Accuracy, MetricsLambda
import torch

def f1(r, p):
    return torch.mean(2 * p * r / (p + r + 1e-20)).item()


class Metrics():

    def __init__(self):

        self.acc = Accuracy(is_multilabel=True)
        self.acc_40 = Accuracy(is_multilabel=True)
        self.acc_60 = Accuracy(is_multilabel=True)

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
        self.acc.update((outputs1, labels_batch))
        self.prec.update((outputs1, labels_batch))
        self.rec.update((outputs1, labels_batch))
        self.prec_f1.update((outputs1, labels_batch))
        self.rec_f1.update((outputs1, labels_batch))

        outputs_40 = (outputs > 0.4)
        self.acc_40.update((outputs_40, labels_batch))
        self.prec_40.update((outputs_40, labels_batch))
        self.rec_40.update((outputs_40, labels_batch))
        self.prec_f1_40.update((outputs_40, labels_batch))
        self.rec_f1_40.update((outputs_40, labels_batch))

        outputs_60 = (outputs > 0.6)
        self.acc_60.update((outputs_60, labels_batch))
        self.prec_60.update((outputs_60, labels_batch))
        self.rec_60.update((outputs_60, labels_batch))
        self.prec_f1_60.update((outputs_60, labels_batch))
        self.rec_f1_60.update((outputs_60, labels_batch))

    def reset(self):

        self.acc.reset()
        self.acc_40.reset()
        self.acc_60.reset()

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

    def compute_valid(self, loss):

        return {'loss': loss,
                 'precision': self.prec.compute(),
                 'precision_40': self.prec_40.compute(),
                 'precision_60': self.prec_60.compute(),
                 'recall': self.rec.compute(),
                 'recall_40': self.prec_40.compute(),
                 'recall_60': self.prec_60.compute(),
                 'f1_score': self.f1_score.compute(),
                 'f1_score_40': self.f1_score_40.compute(),
                 'f1_score_60': self.f1_score_60.compute(),
                }

    def compute_train(self, loss, ep, epoch_time):

        return {'epoch': int(ep),
                'loss': loss,
                'accuracy': self.acc.compute(),
                'accuracy_40': self.acc_40.compute(),
                'accuracy_60': self.acc_60.compute(),
                'precision': self.prec.compute(),
                'precision_40': self.prec_40.compute(),
                'precision_60': self.prec_60.compute(),
                'recall': self.rec.compute(),
                'recall_40': self.prec_40.compute(),
                'recall_60': self.prec_60.compute(),
                'f1_score': self.f1_score.compute(),
                'f1_score_40': self.f1_score_40.compute(),
                'f1_score_60': self.f1_score_60.compute(),
                'epoch_time': epoch_time
                }
