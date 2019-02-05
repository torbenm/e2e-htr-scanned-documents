import tensorflow as tf

from . import Extendable, Executable


class ClassRunner(Executable, Extendable):

    def __init__(self, dataset, logger=None, subset="test", config={}):
        super().__init__(config=config, logger=logger, subset=subset, dataset=dataset)

    def get_logger_prefix(self, epoch):
        return "Classifying"

    def get_batches(self):
        return self.dataset.generateBatch(
            self.config['batch'], self.config['max_batches'], self.subset, True)

    def get_feed_dict(self, batch, graph):
        # X, Y, L, F = batch
        X = batch[0]
        return {
            graph['x']: X,
            graph['l']: [graph['logits'].shape[0]] * len(X)
        }

    def get_graph_executables(self, graph):
        return graph['class_pred']

    def after_iteration(self, batch, execution_results):
        self.transcriptions['class'].extend(execution_results)
        self.transcriptions['original'].extend(batch[1])
        self.transcriptions['files'].extend(batch[3])

    def before_call(self):
        self.transcriptions = {
            'files': [],
            'original': [],
            'class': []
        }

    def will_continue(self, epoch):
        return epoch == 0
