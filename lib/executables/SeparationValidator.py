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
        self.build_sep_precision(graph)
        self.build_sep_recall(graph)
        self.build_sep_accuracy(graph)

    def max_batches(self):
        return self.config.defaultchain(
            'max_batches.sep.val', 'max_batches.sep', 'max_batches')

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], max_batches=self.max_batches(), dataset=self.subset)

    def get_graph_executables(self, graph):
        return [graph['loss'], self._sep_acc, self._sep_rec, self._sep_prec]

    def before_call(self):
        self.all_training_loss = []
        self.ac_total = []
        self.prec_total = []
        self.rec_total = []

    def after_iteration(self, batch, execution_results):
        training_loss, ac, rec, prec = execution_results
        self.all_training_loss.append(training_loss)
        self.ac_total.append(ac)
        self.rec_total.append(rec)
        self.prec_total.append(prec)

    def after_call(self):
        self.training_loss = np.ma.masked_invalid(
            self.all_training_loss).mean()
        self.mean_ac = np.ma.masked_invalid(
            self.ac_total).mean()
        self.mean_rec = np.ma.masked_invalid(
            self.rec_total).mean()
        self.mean_prec = np.ma.masked_invalid(
            self.prec_total).mean()

    def summarize(self, summary):
        summary.update({
            "sep val loss": self.training_loss,
            "sep val ac": self.mean_ac,
            "sep val rec": self.mean_rec,
            "sep val prec": self.mean_prec,
        })
