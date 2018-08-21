import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class ClassValidator(Executable, Extendable):

    validation_results = None

    def __init__(self, **kwargs):
        kwargs.setdefault('subset', 'print_dev')
        super().__init__(**kwargs)
        self.prefix = kwargs.get('prefix', '')

    def get_logger_prefix(self, epoch):
        return "Validating"

    def max_batches(self):
        return self.config.defaultchain(
            'max_batches.class.dev', 'max_batches.class', 'max_batches')

    def extend_graph(self, graph):
        self.build_accuracy(graph)

    def get_feed_dict(self, batch, graph):
        X, Y, _ = batch
        return {
            graph['x']: X,
            graph['class_y']: Y
        }

    def get_graph_executables(self, graph):
        return self._accuracy

    def before_call(self):
        self.acc_total = []

    def after_iteration(self, batch, execution_results):
        self.acc_total.append(execution_results)

    def after_call(self):
        self.validation_results = {
            "accuracy": np.mean(self.acc_total)
        }

    def summarize(self, summary):
        summary.update({
            "{}class acc".format(self.prefix): self.validation_results["accuracy"]
        })
