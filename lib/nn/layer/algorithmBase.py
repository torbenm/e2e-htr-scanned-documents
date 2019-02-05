
import abc
from lib.Configuration import Configuration


"""
{
    "encoder": "resnet", "cnn", "..."
    "cnn": {

    },
    "resnet": {

    },
    "recurrent": {
        "dropout": ...,
        "size": ...,
        "cell": ...
    },
    "decoder": "beam", "greedy"
}
"""


class AlgorithmBase(object):

    __metaclass__ = abc.ABCMeta

    _cpu = False

    def set_cpu(self, is_cpu):
        self._cpu = is_cpu

    def __init__(self, config, defaults):
        self._config = Configuration(config)
        self._defaults = Configuration(defaults)

    def __getitem__(self, key):
        default = self._defaults.default(key, None)
        return self._config.default(key, default)

    @abc.abstractmethod
    def build_graph():
        pass
