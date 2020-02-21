from ignite.metrics import Precision, Recall, Accuracy, MetricsLambda
import torch


def f1(r, p):
    return torch.mean(2 * p * r / (p + r + 1e-20)).item()


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y

class Metrics:
    def __init__(self):

        self.acc = Accuracy(is_multilabel=True)
        self.prec = Precision(average=True, is_multilabel=True)
        self.rec = Recall(average=True, is_multilabel=True)

    def update(self, outputs, labels_batch):

        outputs, labels_batch = thresholded_output_transform((outputs, labels_batch))
        self.acc.update((outputs, labels_batch))
        self.prec.update((outputs, labels_batch))
        self.rec.update((outputs, labels_batch))

    def reset(self):

        self.acc.reset()
        self.prec.reset()
        self.rec.reset()

    def compute_valid(self, loss):
        return {'loss': loss.detach().cpu().numpy(),
                'accuracy': self.acc.compute(),
                'precision': self.prec.compute(),
                'recall': self.rec.compute(),
                }

    def compute_train(self, loss, ep, epoch_time):

        return {'epoch': int(ep),
                'loss': loss.detach().cpu().numpy(),
                'accuracy': self.acc.compute(),
                'precision': self.prec.compute(),
                'recall': self.rec.compute(),
                'epoch_time': epoch_time
                }
