import abc


class Dataset(object):

    __metaclass__ = abc.ABCMeta
    requires_splitting = False

    @abc.abstractmethod
    def getFilesAndTruth(self, subset):
        pass

    @abc.abstractmethod
    def getDownloadInfo(self):
        pass

    @abc.abstractmethod
    def getIdentifier(self):
        pass

    def do_split(self, raw_path):
        pass
