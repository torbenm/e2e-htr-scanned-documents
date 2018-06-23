from nn.layer.Layer import Layer
from nn.layer import resnet
from nn.util import log_1d

DEFAULTS = {
    "size": 3,
    "bottleneck": False,
    "num_filters": 4,
    "kernel_size": 3,
    "conv_stride": 1,
    "first_pool_size": 2,
    "first_pool_stride": 2,
    "block_sizes": [1] * 3,
    "block_strides": [1] * 3
}


class ResNetEncoder(Layer):

    def __init__(self, config, defaults=None, data_format='nhwc'):
        super(Layer, self).__init__(config, defaults or DEFAULTS, data_format)

    def __call__(self, x, is_train):
        ResNet = resnet.Model(resnet_size=self['resnet_size'], bottleneck=self['bottleneck'], num_filters=self['num_filters'],
                              kernel_size=self['kernel_size'],
                              conv_stride=self['conv_stride'], first_pool_size=self[
                                  'first_pool_size'], first_pool_stride=self['first_pool_stride'],
                              block_sizes=self['block_sizes'], block_strides=self['block_strides'], data_format=self._parse_format())
        return log_1d(ResNet(x, is_train))
