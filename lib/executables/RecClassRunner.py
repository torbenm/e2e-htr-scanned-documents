import tensorflow as tf

from . import Extendable, Executable


class RecClassRunner(Executable, Extendable):

    def __init__(self, dataset, logger=None, subset="test", config={}):
        super().__init__(config=config, logger=logger, subset=subset, dataset=dataset)

    def get_logger_prefix(self, epoch):
        return "Transcribing"

    def extend_graph(self, graph):
        self.build_decoded_dense(graph)

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
        return [self._decoded_dense, graph['class_pred']]

    def after_iteration(self, batch, execution_results):
        results_, class_ = execution_results
        self.transcriptions['class'].extend(class_)
        self.transcriptions['files'].extend(batch[3])
        self.transcriptions['trans'].extend(results_)

    def before_call(self):
        self.transcriptions = {
            'files': [],
            'trans': [],
            'class': []
        }

    def will_continue(self, epoch):
        return epoch == 0
