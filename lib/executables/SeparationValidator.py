import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class SeparationValidator(Executable, Extendable):

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

    def extend_graph(self, graph):
        self.build_tp(graph)
        self.build_tn(graph)
        self.build_sep_accuracy(graph)

    def max_batches(self):
        return self.config.defaultchain(
            'max_batches.sep.val', 'max_batches.sep', 'max_batches')

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], max_batches=self.max_batches(), dataset=self.subset)

    def get_graph_executables(self, graph):
        return [graph['loss'], self._sep_acc, self._tp, self._tn]

    def before_call(self):
        self.all_training_loss = []
        self.ac_total = []
        self.tp_total = []
        self.tn_total = []

    def after_iteration(self, batch, execution_results):
        training_loss, ac, tp, tn = execution_results
        self.all_training_loss.append(training_loss)
        self.ac_total.append(ac)
        self.tp_total.append(tp)
        self.tn_total.append(tn)

    def after_call(self):
        self.training_loss = np.ma.masked_invalid(
            self.all_training_loss).mean()
        self.mean_ac = np.ma.masked_invalid(
            self.ac_total).mean()
        self.mean_tp = np.ma.masked_invalid(
            self.tp_total).mean()
        self.mean_tn = np.ma.masked_invalid(
            self.tn_total).mean()

    def summarize(self, summary):
        summary.update({
            "sep val loss": self.training_loss,
            "sep val ac": self.mean_ac,
            "sep val tp": self.mean_tp,
            "sep val tn": self.mean_tn,
        })
