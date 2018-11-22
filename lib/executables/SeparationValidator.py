import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class SeparationValidator(Executable, Extendable):

    training_loss = 0

    def __init__(self, **kwargs):
        kwargs.setdefault('subset', 'dev')
        self.prefix = kwargs.get('prefix', 'dev')
        self.exit_afterwards = kwargs.get('exit_afterwards', False)
        super().__init__(**kwargs)

    def get_logger_prefix(self, epoch):
        return "Validating {}".format(self.prefix)

    def get_feed_dict(self, batch, graph):
        X, Y, _ = batch
        return {
            graph['x']: X,
            graph['y']: Y,
        }

    def extend_graph(self, graph):
        # self.build_sep_precision(graph)
        # self.build_sep_recall(graph)
        # self.build_sep_fmeasure(graph)
        self.build_tp(graph)
        self.build_fn(graph)
        self.build_tn(graph)

    def max_batches(self):
        return self.config.defaultchain(
            'max_batches.sep.val', 'max_batches.sep', 'max_batches')

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], max_batches=self.max_batches(), dataset=self.subset)

    def get_graph_executables(self, graph):
        return [graph['loss'], self._tn, self._fn, self._tp]

    def before_call(self):
        self.all_training_loss = []
        self.tp = 0
        self.fn = 0
        self.tn = 0
        # self.f_total = []
        # self.prec_total = []
        # self.rec_total = []

    def after_iteration(self, batch, execution_results):
        training_loss, tn, fn, tp = execution_results
        # training_loss, f, rec, prec = execution_results
        self.all_training_loss.append(training_loss)
        self.tp += tp
        self.fn += fn
        self.tn += tn
        # self.f_total.append(f)
        # self.rec_total.append(rec)
        # self.prec_total.append(prec)

    def after_call(self):
        self.training_loss = np.ma.masked_invalid(
            self.all_training_loss).mean()
        # self.mean_f = np.ma.masked_invalid(
        #     self.f_total).mean()
        # self.mean_rec = np.ma.masked_invalid(
        #     self.rec_total).mean()
        # self.mean_prec = np.ma.masked_invalid(
        #     self.prec_total).mean()
        self.mean_rec = self.tp / (self.tp+self.fn)
        self.mean_prec = self.tp / (self.tp+self.fp)
        self.mean_f = 2.0 * (self.mean_prec * self.mean_rec) / \
            (self.mean_prec + self.mean_rec)

    def summarize(self, summary):
        summary.update({
            "sep {} loss".format(self.prefix): self.training_loss,
            "sep {} f".format(self.prefix): self.mean_f,
            "sep {} rec".format(self.prefix): self.mean_rec,
            "sep {} prec".format(self.prefix): self.mean_prec,
        })

    def will_continue(self, epoch):
        if self.exit_afterwards:
            return epoch == 0
        else:
            return True
