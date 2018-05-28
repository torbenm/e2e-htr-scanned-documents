
import abc


class AlgorithmBase(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def build_graph():
        pass
