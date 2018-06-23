
import abc
from config.config import Configuration


class Layer(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, config, defaults, data_format='nhwc'):
        self._config = Configuration(config)
        self._defaults = Configuration(defaults)
        self._format = data_format

    def __getitem__(self, key):
        default = self._defaults.default(key, None)
        return self._config.default(key, default)

    def _parse_format():
        return 'channels_first' if self._format == 'nchw' else 'channels_last'

    @abc.abstractmethod
    def __call__(self, x, is_train):
        pass
