import tensorflow as tf
import os

from .Executable import Executable
from lib.util.file import writeJson, readJson


class Saver(Executable):

    def __init__(self, foldername, every_epoch=1):
        super().__init(every_epoch=every_epoch)
        self.foldername = foldername
        self.saver = tf.train.Saver(max_to_keep=None)

    def __call__(self, executor, epoch, session, graph):
        saver.save(session, self.foldername, global_step=epoch)

    def _copy_config(self):
        # Copy configurations to models folder
        os.makedirs(self.foldername, exist_ok=True)
        self.config.save(self.foldername, 'algorithm')
        self.dataset.meta.save(self.foldername, 'data_meta')
        self.config.save(self.foldername, 'algorithm')
        writeJson(os.path.join(self.foldername,
                               'vocab.json'), self.dataset.vocab)
