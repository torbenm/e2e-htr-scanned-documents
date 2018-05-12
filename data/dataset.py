import abc


class Dataset(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generateEpochs(self, batch_size, num_epochs):
        pass
