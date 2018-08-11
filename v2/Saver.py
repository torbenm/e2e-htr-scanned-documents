import tensorflow as tf

from .Executable import Executable


class Saver(Executable):

    def __init__(self, foldername, every_epoch=1):
        self.every_epoch = every_epoch
        self.foldername = foldername
        self.saver = tf.train.Saver(max_to_keep=None)

    def __call__(self, executor, epoch, session, graph):
        saver.save(session, self.foldername, global_step=epoch)

    def will_run(self, epoch):
        return (epoch + 1) % self.every_epoch == 0
