import tensorflow as tf

from . import Extendable, Executable


class RecognitionRunner(Executable, Extendable):

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
        X = batch[0]
        return {
            graph['x']: X,
            graph['l']: [graph['logits'].shape[0]] * len(X)
        }

    def get_graph_executables(self, graph):
        return self._decoded_dense

    def after_iteration(self, batch, execution_results):
        decoded_ = execution_results
        self.transcriptions['files'].extend(batch[3])
        self.transcriptions['trans'].extend(decoded_)

    def before_call(self):
        self.transcriptions = {
            'files': [],
            'trans': []
        }

    def will_continue(self, epoch):
        return epoch == 0
