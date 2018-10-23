import tensorflow as tf
import numpy as np
import time
from typing import Dict

from . import Extendable, Executable
from lib.Configuration import Configuration
from lib.Logger import Logger


class Executable(object):

    def __init__(self, **kwargs):
        self.subset = kwargs.get('subset', 'train')
        self.logger = kwargs.get('logger', None)
        self.every_epoch = kwargs.get('every_epoch', 1)
        self.dataset = kwargs.get('dataset', None)
        self.config = Configuration(kwargs.get('config', {}))

    def __call__(self, executor, epoch, session, graph):
        start_time = time.time()
        self.before_call()
        self.dataset.before_epoch()
        step = 0
        total_steps = self.get_batch_count()
        self.log_progress(epoch, 0, total_steps)
        for batch in self.get_batches():
            step += 1
            feed_dict = self.get_feed_dict(batch, graph)
            execution_results = session.run(
                self.get_graph_executables(graph), feed_dict)
            self.after_iteration(batch, execution_results)
            self.log_progress(epoch, step, total_steps)
        self.after_call()
        self.execution_time = time.time() - start_time

    def will_continue(self, epoch: int) -> bool:
        return True

    def will_run(self, epoch):
        return (epoch + 1) % self.every_epoch == 0

    def denseNDArrayToSparseTensor(self, arr, sparse_val=-1):
        idx = np.where(arr != sparse_val)
        return tf.SparseTensorValue(np.vstack(idx).T, arr[idx], arr.shape)

    def max_batches(self):
        return self.config['max_batches']

    def get_batch_count(self):
        return self.dataset.getBatchCount(
            self.config['batch'], self.max_batches(), self.subset)

    def log_progress(self, epoch, step, total):
        if self.logger is not None:
            self.logger.progress(self.get_logger_prefix(epoch), step, total)

    # TYPICAL OVERRIDES
    def extend_graph(self, graph):
        pass

    def get_batches(self):
        return self.dataset.generateBatch(self.config['batch'], self.max_batches(), self.subset)

    def get_logger_prefix(self, epoch):
        return "Undefined"

    def get_feed_dict(self, batch, graph):
        return {}

    def get_graph_executables(self, graph):
        return []

    def before_call(self):
        pass

    def after_call(self):
        pass

    def after_iteration(self, batch, execution_results):
        pass

    def summarize(self, summary):
        pass
