import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class SeparationTrainer(Executable, Extendable):

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
        self.build_sep_recall(graph)
        self.build_sep_precision(graph)
        self.build_sep_fmeasure(graph)

    def max_batches(self):
        return self.config.defaultchain(
            'max_batches.sep.train', 'max_batches.sep', 'max_batches')

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], max_batches=self.max_batches(), dataset=self.subset, augmentable=True)

    def get_graph_executables(self, graph):
        return [graph['loss'], graph['train_step'], self._sep_f, self._sep_rec, self._sep_prec, graph['gradients']]

    def before_call(self):
        self.all_training_loss = []
        self.rec_total = []
        self.prec_total = []
        self.f_total = []

    def after_iteration(self, batch, execution_results):
        training_loss, _, f, rec, prec, _ = execution_results
        self.all_training_loss.append(training_loss)
        self.rec_total.append(rec)
        self.prec_total.append(prec)
        self.f_total.append(f)

    def after_call(self):
        self.training_loss = np.ma.masked_invalid(
            self.all_training_loss).mean()
        self.mean_rec = np.ma.masked_invalid(
            self.rec_total).mean()
        self.mean_prec = np.ma.masked_invalid(
            self.prec_total).mean()
        self.mean_f = np.ma.masked_invalid(
            self.f_total).mean()

    def summarize(self, summary):
        summary.update({
            "sep loss": self.training_loss,
            "sep f": self.mean_f,
            "sep rec": self.mean_rec,
            "sep prec": self.mean_prec,
        })
