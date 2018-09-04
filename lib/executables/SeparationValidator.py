import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class SeparationValidator(Executable):

    training_loss = 0

    def __init__(self, **kwargs):
        kwargs.setdefault('subset', 'dev')
        super().__init__(**kwargs)

    def get_logger_prefix(self, epoch):
        return "Validating"

    def get_feed_dict(self, batch, graph):
        X, Y, _ = batch
        return {
            graph['x']: X,
            graph['y']: Y,
        }

    def max_batches(self):
        return self.config.defaultchain(
            'max_batches.sep.val', 'max_batches.sep', 'max_batches')

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], max_batches=self.max_batches(), dataset=self.subset)

    def get_graph_executables(self, graph):
        return graph['loss']

    def before_call(self):
        self.all_training_loss = []

    def after_iteration(self, batch, execution_results):
        training_loss = execution_results
        self.all_training_loss.append(training_loss/len(batch[0]))

    def after_call(self):
        self.training_loss = np.ma.masked_invalid(
            self.all_training_loss).mean()

    def summarize(self, summary):
        summary.update({
            "sep val loss": self.training_loss
        })
