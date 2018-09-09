import tensorflow as tf
import numpy as np
from nn.layer.algorithmBaseV2 import AlgorithmBaseV2
from nn.util import log_1d
from nn.layer.histogrammed import conv2d, batch_normalization
from nn.layer.tf_unet import unet

DEFAULTS = {}


class TFUnet(AlgorithmBaseV2):

    def __init__(self, config):
        super(Unet, self).__init__(config, DEFAULTS)

    def configure(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.channels = kwargs.get('channels', 1)
        self.slice_width = kwargs.get('slice_width', 320)
        self.slice_height = kwargs.get('slice_height', 320)
        self.class_weights = kwargs.get('class_weights', None)
        self.n_class = kwargs.get('n_class', 2)

    def build_graph(self):
        net = unet.Unet(channels=self.channels,
                        n_class=self.n_class,
                        layers=3,
                        features_root=64,
                        cost_kwargs=dict(regularizer=0.001),
                        )
        return dict(
            x=x,
            y=y,
            is_train=is_train,
            output=output,
            loss=loss,
            train_step=train_step
        )
