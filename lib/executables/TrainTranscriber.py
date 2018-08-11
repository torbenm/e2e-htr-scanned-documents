import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class TrainTranscriber(Executable):

    training_loss = 0

    def __init__(self, **kwargs):
        kwargs.setdefault('subset', 'train')
        super().__init__(**kwargs)

    def get_logger_prefix(self, epoch):
        return "Epoch {:4d}".format(epoch)

    def get_feed_dict(self, batch, graph):
        X, Y, _ = batch
        return {
            graph['x']: X,
            graph['y']: denseNDArrayToSparseTensor(Y),
            graph['l']: [self.dataset.max_length] * len(X),
            graph['is_train']: True
        }

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], max_batches=self.config['max_batches'], dataset=self.subset, augmentable=True)

    def get_graph_executables(self, graph):
        return [graph['total_loss'], graph['train_step']]

    def before_call(self):
        self.all_training_loss = []

    def after_iteration(self, batch, execution_results):
        training_loss, _ = execution_results
        self.all_training_loss.extend(training_loss)

    def after_call(self):
        self.training_loss = np.ma.masked_invalid(
            self.all_training_loss).mean()

    def summarize(self, summary):
        summary.update({
            "trans loss": self.training_loss
        })
