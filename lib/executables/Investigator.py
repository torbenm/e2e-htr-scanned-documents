import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class Investigator(Executable, Extendable):

    training_loss = 0

    def __init__(self, **kwargs):
        kwargs.setdefault('subset', 'test')
        super().__init__(**kwargs)

    def extend_graph(self):
        self.build_decoded_dense(graph)
        self.build_cer(graph)

    def get_logger_prefix(self, epoch):
        return "Investigator".format(epoch)

    def get_feed_dict(self, batch, graph):
        X, Y, _, _ = batch
        return {
            graph['x']: X,
            graph['y']: self.denseNDArrayToSparseTensor(Y),
            graph['l']: [self.dataset.max_length] * len(X)
            # graph['l']: [graph['logits'].shape[0]] * len(X)
        }

    def get_batches(self):
        return self.dataset.generateBatch(self.config['batch'], self.config['max_batches'], self.subset, True)

    def get_graph_executables(self, graph):
        runners = [self._decoded_dense, self._cer]
        if self.dataset.meta.default('printed', False):
            runners.append(graph['class_pred'])
        return runners

    def before_call(self):
        self.transcriptions = {
            'files': [],
            'trans': [],
            'class': [],
            'cer': [],
            'truth': []
        }

    def after_iteration(self, batch, execution_results):
        if self.dataset.meta.default('printed', False):
            results_, cer_, class_ = execution_results
            self.transcriptions['class'].extend(class_)
        else:
            results_, cer_ = execution_results
        self.transcriptions['files'].extend(batch[3])
        self.transcriptions['cer'].extend(cer_)
        self.transcriptions['truth'].extend(batch[1])
        self.transcriptions['trans'].extend(results_)
