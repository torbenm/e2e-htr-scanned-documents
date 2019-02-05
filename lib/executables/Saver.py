import tensorflow as tf
import os

from .Executable import Executable
from lib.util.file import writeJson, readJson


class Saver(Executable):

    execution_time = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.foldername = kwargs.get('foldername')
        self._copy_config()

    def __call__(self, executor, epoch, session, graph):
        tf.train.Saver(max_to_keep=None).save(
            session, os.path.join(self.foldername, 'model'), global_step=epoch)

    def _copy_config(self):
        # Copy configurations to models folder
        os.makedirs(self.foldername, exist_ok=True)
        self.config.save(self.foldername, 'algorithm')
        self.dataset.meta.save(self.foldername, 'data_meta')
        self.config.save(self.foldername, 'algorithm')
        writeJson(os.path.join(self.foldername,
                               'vocab.json'), self.dataset.vocab)
