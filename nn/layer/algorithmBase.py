
import abc


class AlgorithmBase(object):

    __metaclass__ = abc.ABCMeta

    _cpu = False

    def set_cpu(self, is_cpu):
        self._cpu = is_cpu

    @abc.abstractmethod
    def build_graph():
        pass
