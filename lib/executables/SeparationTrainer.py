import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class SeparationTrainer(Executable):

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
            graph['y']: Y,
            graph['is_train']: True
        }

    def extend_graph(self, graph):
        self.build_tp(graph)
        self.build_fp(graph)
        # self.build_tn(graph)
        # self.build_fn(graph)

    def max_batches(self):
        return self.config.defaultchain(
            'max_batches.sep.train', 'max_batches.sep', 'max_batches')

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], max_batches=self.max_batches(), dataset=self.subset)

    def get_graph_executables(self, graph):
        return [graph['loss'], graph['train_step'], self._tp, self._fp]

    def before_call(self):
        self.all_training_loss = []
        self.tp_total = []
        self.fp_total = []

    def after_iteration(self, batch, execution_results):
        training_loss, _, tp, fp = execution_results
        self.all_training_loss.append(training_loss)

    def after_call(self):
        self.training_loss = np.ma.masked_invalid(
            self.all_training_loss).mean()
        self.mean_tp = np.ma.masked_invalid(
            self.tp_total).mean()
        self.mean_fp = np.ma.masked_invalid(
            self.fp_total).mean()

    def summarize(self, summary):
        summary.update({
            "sep loss": self.training_loss,
            "sep tp": self.mean_tp,
            "sep fp": self.mean_fp,
        })
