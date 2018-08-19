import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class RecognitionValidator(Executable, Extendable):

    validation_results = None

    def __init__(self, **kwargs):
        kwargs.setdefault('subset', 'dev')
        super().__init__(**kwargs)
        self.prefix = kwargs.get('prefix', '')

    def get_logger_prefix(self, epoch):
        return "Validating"

    def extend_graph(self, graph):
        self.build_cer(graph)
        self.build_decoded_dense(graph)

    def get_feed_dict(self, batch, graph):
        X, Y, _ = batch
        return {
            graph['x']: X,
            graph['y']: self.denseNDArrayToSparseTensor(Y),
            graph['l']: [self.dataset.max_length] * len(X)
        }

    def get_graph_executables(self, graph):
        return [self._decoded_dense, self._cer, graph['total_loss']]

    def before_call(self):
        self.cer_total = []
        self.loss_total = []
        self.examples = {
            'Y': [],
            'trans': []
        }

    def after_iteration(self, batch, execution_results):
        trans_, cer_, loss_ = execution_results
        self.cer_total.extend(cer_)
        self.loss_total.extend(loss_)
        self.examples['trans'].extend(trans_)
        self.examples['Y'].extend(batch[1])

    def after_call(self):
        self.validation_results = {
            'examples': self.examples,
            'loss': np.ma.masked_invalid(self.loss_total).mean(),
            'cer': np.mean(self.cer_total)
        }

    def summarize(self, summary):
        summary.update({
            "{}cer".format(self.prefix): self.validation_results["cer"]
        })
