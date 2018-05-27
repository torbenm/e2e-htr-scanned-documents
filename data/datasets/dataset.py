import abc


class Dataset(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getFilesAndTruth(self, subset):
        pass

    @abc.abstractmethod
    def getDownloadInfo(self):
        pass

    @abc.abstractmethod
    def getIdentifier(self):
        pass
