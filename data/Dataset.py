import abc


class Dataset(object):
    __metaclass__ = abc.ABCMeta

    max_length = 0
    channels = 1

    @abc.abstractmethod
    def info(self):
        pass

    @abc.abstractmethod
    def compile(self, text):
        pass

    @abc.abstractmethod
    def decompile(self, values):
        pass

    @abc.abstractmethod
    def generateBatch(self, batch_size, max_batches=0, dataset="train", with_filepath=False):
        pass

    @abc.abstractmethod
    def generateEpochs(self, batch_size, num_epochs, max_batches=0, dataset="train", with_filepath=False):
        pass

    @abc.abstractmethod
    def getBatchCount(self, batch_size, max_batches=0, dataset="train"):
        pass

    def before_epoch(self):
        pass
