import abc


class Dataset(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generateEpochs(self, batch_size, num_epochs):
        pass

    @abc.abstractmethod
    def prepareDataset(self, validation_size=0, test_size=0, shuffle=False):
        pass

    @abc.abstractmethod
    def getValidationSet(self):
        pass

    @abc.abstractmethod
    def compile(self, text):
        pass

    @abc.abstractmethod
    def decompile(self, array):
        pass

    @abc.abstractmethod
    def getBatchCount(self):
        pass
