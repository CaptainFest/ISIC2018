#from ignite.metrics import Precision, Recall, Accuracy, MetricsLambda
from tensorflow.keras.metrics import Accuracy, Precision, Recall
import tensorflow as tf
import torch

def f1(r, p):
    return torch.mean(2 * p * r / (p + r + 1e-20)).item()


class Metrics:

    def __init__(self):

        self.acc = Accuracy()
        self.prec = Precision()
        self.rec = Recall()

    def update(self, outputs, labels_batch):

        outputs1 = (outputs > 0.5)
        self.acc.update_state(outputs1, labels_batch)
        self.prec.update_state(outputs1, labels_batch)
        self.rec.update_state(outputs1, labels_batch)

    def reset(self):

        self.acc.reset_states()
        self.prec.reset_states()
        self.rec.reset_states()

    def compute_valid(self, loss):

        return {'loss': loss.detach().cpu().numpy(),
                'accuracy': self.acc.result().cpu().numpy(),
                'precision': self.prec.result().cpu().numpy(),
                'recall': self.rec.result().cpu().numpy()
                }

    def compute_train(self, loss, ep, epoch_time):


        return {'epoch': int(ep),
                'loss': loss.detach().cpu().numpy(),
                'accuracy': self.acc.result().eval(session=tf.compat.v1.Session()),
                'precision': self.prec.result().eval(session=tf.compat.v1.Session()),
                'recall': self.rec.result().eval(session=tf.compat.v1.Session()),
                'epoch_time': epoch_time
                }
