import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class SeparationValidator(Executable):

    training_loss = 0

    _tp = None
    _tn = None
    _fn = None
    _fp = None
    _pred_res = None
    _y_res = None

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
        self.build_tn(graph)
        self.build_fn(graph)

    def build_tp(self, graph):
        if self._tp is None:
            pred_res = self.build_pred_res(graph)
            y_res = self.build_y_res(graph)
            self._tp = tf.reduce_mean(
                tf.cast(tf.equal(tf.boolean_mask(pred_res, tf.equal(y_res, 0)), 0), tf.float32))
        return self._tp

    def build_fp(self, graph):
        if self._fp is None:
            pred_res = self.build_pred_res(graph)
            y_res = self.build_y_res(graph)
            self._fp = tf.reduce_mean(
                tf.cast(tf.equal(tf.boolean_mask(pred_res, tf.equal(y_res, 1)), 0), tf.float32))
        return self._fp

    def build_fn(self, graph):
        if self._fn is None:
            pred_res = self.build_pred_res(graph)
            y_res = self.build_y_res(graph)
            self._fn = tf.reduce_mean(
                tf.cast(tf.equal(tf.boolean_mask(pred_res, tf.equal(y_res, 0)), 1), tf.float32))
        return self._fn

    def build_tn(self, graph):
        if self._tn is None:
            pred_res = self.build_pred_res(graph)
            y_res = self.build_y_res(graph)
            self._tn = tf.reduce_mean(
                tf.cast(tf.equal(tf.boolean_mask(pred_res, tf.equal(y_res, 1)), 1), tf.float32))
        return self._tn

    def build_pred_res(self, graph):
        if self._pred_res is None:
            self._pred_res = tf.argmax(graph['output'], 3)
        return self._pred_res

    def build_y_res(self, graph):
        if self._y_res is None:
            _y = tf.cast(tf.reshape(graph['y']/tf.constant(255.0),
                                    [-1, graph['y'].shape[1], graph['y'].shape[2]]), tf.int32)
            _y = tf.one_hot(_y, 2)
            self._y_res = tf.argmax(_y, 3)
        return self._y_res

    def max_batches(self):
        return self.config.defaultchain(
            'max_batches.sep.val', 'max_batches.sep', 'max_batches')

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], max_batches=self.max_batches(), dataset=self.subset)

    def get_graph_executables(self, graph):
        return [graph['loss'], self._tp, self._fn, self._tn, self._fp]

    def before_call(self):
        self.all_training_loss = []
        self.tn_total = []
        self.fn_total = []
        self.tp_total = []
        self.fp_total = []

    def after_iteration(self, batch, execution_results):
        training_loss, tp, fn, tn, fp = execution_results
        self.all_training_loss.append(training_loss)
        self.tn_total.append(tn)
        self.tp_total.append(tp)
        self.fp_total.append(fp)
        self.fn_total.append(fn)

    def after_call(self):
        self.training_loss = np.ma.masked_invalid(
            self.all_training_loss).mean()
        self.mean_tn = np.ma.masked_invalid(
            self.tn_total).mean()
        self.mean_fn = np.ma.masked_invalid(
            self.fn_total).mean()
        self.mean_tp = np.ma.masked_invalid(
            self.tp_total).mean()
        self.mean_fp = np.ma.masked_invalid(
            self.fp_total).mean()

    def summarize(self, summary):
        summary.update({
            "sep val loss": self.training_loss,
            "sep tn": self.mean_tn,
            "sep tp": self.mean_tp,
            "sep fp": self.mean_fp,
            "sep fn": self.mean_fn,
        })
