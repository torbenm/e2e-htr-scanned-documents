import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class DnCNNRunner(Executable):

    outputs = []

    def __init__(self, **kwargs):
        kwargs.setdefault('subset', 'train')
        super().__init__(**kwargs)
        self.after_iter = kwargs.get('after_iter')

    def get_logger_prefix(self, epoch):
        return "Epoch {:4d}".format(epoch)

    def get_feed_dict(self, batch, graph):
        X, _, _ = batch
        return {
            graph['x']: X
        }

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], max_batches=self.config['max_batches'], dataset=self.subset)

    def get_graph_executables(self, graph):
        return graph['output']

    def before_call(self):
        self.outputs = []

    def after_iteration(self, batch, execution_results):
        output = execution_results
        self.outputs.extend(output)
        if self.after_iter is not None:
            self.after_iter(output, batch[0])

    def after_call(self):
        pass

    def summarize(self, summary):
        pass

    def will_continue(self, epoch):
        return epoch == 0
