import tensorflow as tf

from .Executable import Executable
from .Extendable import Extendable


class Transcriber(Executable, Extendable):

    transcriptions = None
    extended = False

    def __init__(self, dataset, logger=None, subset="test", config={}):
        self.every_epoch = every_epoch
        self.dataset = dataset
        self.subset = subset
        self.logger = logger
        self.config = config

    def extend_graph(self, graph):
        # Will do this on the fly!
        pass

    def __call__(self, executor, epoch, session, graph):
        transcriptions = {
            'files': [],
            'trans': [],
            'class': []
        }
        # ADDITIONAL GRAPHs
        results = self._build_decoded_dense(graph)
        class_pred = self._build_pred_thresholding(graph)

        # VARIABLES
        steps = 0
        total_steps = self.dataset.getBatchCount(
            self.config['batch'], self.config['max_batches'], subset)

        for X, Y, L, F in self.dataset.generateBatch(self.config['batch'], self.config['max_batches'], subset, True):
            steps += 1
            feed_dict = {
                graph['x']: X,
                graph['l']: [graph['logits'].shape[0]] * len(X)
            }
            results_, class_ = session.run(
                [results, graph['class_pred']], feed_dict)
            transcriptions['class'].extend(class_)
            transcriptions['files'].extend(F)
            transcriptions['trans'].extend(results_)

            if self.logger is not None:
                self.logger.progress("Transcribing", steps, total_steps)
        self.transcriptions = transcriptions

    def continue(self, epoch):
        return epoch == 0
