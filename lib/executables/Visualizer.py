import tensorflow as tf
import numpy as np

from . import Extendable, Executable


class Visualizer(Executable):

    activations = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image = kwargs.get('image')

    def get_logger_prefix(self, epoch):
        return "Running Visualizer"

    def get_feed_dict(self, X):
        return {
            graph['x']: X,
            graph['l']: [graph['logits'].shape[0]] * len(X)
        }

    def get_batch_count(self):
        return 1

    def get_batches(self):
        return [self.dataset.load_image(self.image)]

    def get_graph_executables(self, graph):
        return graph['viz']

    def after_iteration(self, batch, execution_results):
        self.activations = execution_results

    def will_continue(self, epoch: int) -> bool:
        return epoch == 0
