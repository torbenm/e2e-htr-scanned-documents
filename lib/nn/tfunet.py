import tensorflow as tf
import numpy as np
from lib.nn.layer.algorithmBase import AlgorithmBase
from lib.nn.util import log_1d
from lib.nn.layer.histogrammed import conv2d, batch_normalization
from lib.nn.layer.unet import unet
from lib.nn.layer.unet.layers import pixel_wise_softmax

DEFAULTS = {
    "layers": 5,
    "features_root": 16,
    "dropout": 0.5,
    "padding": "SAME",
    "filter_size": 3,
    "pool_size": 2,
    "batch_norm": False,
    "group_norm": False,
    "dropout_enabled": True,
    "cost": {
        "type": "cross_entropy",
        "class_weights": None,
        "regularizer": 0.001
    }
}


class TFUnet(AlgorithmBase):

    def __init__(self, config):
        super(TFUnet, self).__init__(config, DEFAULTS)
        self.viz = None

    def configure(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.channels = kwargs.get('channels', 1)
        self.slice_width = kwargs.get('slice_width', 320)
        self.slice_height = kwargs.get('slice_height', 320)
        self.class_weights = kwargs.get('class_weights', None)
        self.n_class = kwargs.get('n_class', 2)

    def _get_cost(self, logits, y, variables, cost_name, cost_kwargs):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are:
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """

        with tf.name_scope("cost"):
            flat_logits = tf.reshape(logits, [-1, self.n_class])
            flat_labels = tf.reshape(y, [-1, self.n_class])
            if cost_name == "cross_entropy":
                class_weights = cost_kwargs.pop("class_weights", None)

                if class_weights is not None:
                    class_weights = tf.constant(
                        np.array(class_weights, dtype=np.float32))

                    weight_map = tf.multiply(flat_labels, class_weights)
                    weight_map = tf.reduce_sum(weight_map, axis=1)

                    loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                          labels=flat_labels)
                    weighted_loss = tf.multiply(loss_map, weight_map)

                    loss = tf.reduce_mean(weighted_loss)

                else:
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
                                                                                     labels=flat_labels))
            elif cost_name == "dice_coefficient":
                eps = 1e-5
                prediction = pixel_wise_softmax(logits)
                intersection = tf.reduce_sum(prediction * y)
                union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y)
                loss = -(2 * intersection / (union))

            else:
                raise ValueError("Unknown cost function: " % cost_name)

            regularizer = cost_kwargs.pop("regularizer", None)
            if regularizer is not None:
                regularizers = sum([tf.nn.l2_loss(variable)
                                    for variable in variables])
                loss += (regularizer * regularizers)

            return loss

    def _get_optimizer(self, loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate).minimize(loss)
            return optimizer

    def build_graph(self):
        x = tf.placeholder(
            "float", shape=[None, None, None, self.channels], name="x")
        y = tf.placeholder(
            "float", shape=[None, None, None, self.n_class], name="y")
        is_train = tf.placeholder_with_default(False, (), name='is_train')

        logits, variables, viz = unet.create_conv_net(
            x,
            self['dropout'],
            self.channels,
            self.n_class,
            is_train=is_train,
            padding=self['padding'],
            layers=self['layers'],
            features_root=self['features_root'],
            filter_size=self['filter_size'],
            batch_norm=self['batch_norm'],
            group_norm=self['group_norm'],
            with_dropout=self['dropout_enabled'],
            pool_size=self['pool_size'],
            return_viz=True)

        loss = self._get_cost(logits,  y, variables, self['cost.type'], dict(
            regularizer=self['cost.regularizer'], class_weights=self['cost.class_weights']))

        gradients_node = tf.gradients(loss, variables)

        with tf.name_scope("results"):
            output = pixel_wise_softmax(logits)

        train_step = self._get_optimizer(loss)
        return dict(
            x=x,
            y=y,
            viz=viz,
            is_train=is_train,
            output=output,
            loss=loss,
            train_step=train_step,
            gradients=gradients_node
        )
