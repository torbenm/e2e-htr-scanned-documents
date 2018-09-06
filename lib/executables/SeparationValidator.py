import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class SeparationValidator(Executable, Extendable):

    training_loss = 0

    tp_total = []
    tn_total = []
    fn_total = []
    fp_total = []

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
        self.build_fp(graph)
        # self.build_tn(graph)
        # self.build_fn(graph)

    def max_batches(self):
        return self.config.defaultchain(
            'max_batches.sep.val', 'max_batches.sep', 'max_batches')

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], max_batches=self.max_batches(), dataset=self.subset)

    def get_graph_executables(self, graph):
        # return [graph['loss'], self._tp, self._fn, self._tn, self._fp]
        return [graph['loss'], self._tp, self._fp]

    def before_call(self):
        self.all_training_loss = []
        # self.tn_total = []
        # self.fn_total = []
        self.tp_total = []
        self.fp_total = []

    def after_iteration(self, batch, execution_results):
        # training_loss, tp, fn, tn, fp = execution_results
        training_loss, tp, fp = execution_results
        self.all_training_loss.append(training_loss)
        # self.tn_total.append(tn)
        self.tp_total.append(tp)
        self.fp_total.append(fp)
        # self.fn_total.append(fn)

    def after_call(self):
        self.training_loss = np.ma.masked_invalid(
            self.all_training_loss).mean()
        # self.mean_tn = np.ma.masked_invalid(
        #     self.tn_total).mean()
        # self.mean_fn = np.ma.masked_invalid(
        #     self.fn_total).mean()
        self.mean_tp = np.ma.masked_invalid(
            self.tp_total).mean()
        self.mean_fp = np.ma.masked_invalid(
            self.fp_total).mean()

    def summarize(self, summary):
        summary.update({
            "sep val loss": self.training_loss,
            # "sep tn": self.mean_tn,
            "sep val tp": self.mean_tp,
            "sep val fp": self.mean_fp,
            # "sep fn": self.mean_fn,
        })
