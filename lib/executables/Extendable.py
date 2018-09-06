import tensorflow as tf
from lib.Configuration import Configuration


class Extendable(object):

    _decoded_dense = None
    _decoder = None
    _pred_thresholded = None
    _cer = None
    _accuracy = None

    _tp = None
    _tn = None
    _fn = None
    _fp = None
    _pred_res = None
    _y_res = None

    def __init__(self, **kwargs):
        self.config = Configuration(kwargs.get('config', {}))

    def build_decoded_dense(self, graph):
        if self._decoded_dense is None:
            decoded = self.build_decoder(graph)
            self._decoded_dense = tf.sparse_to_dense(
                decoded[0].indices, decoded[0].dense_shape, decoded[0].values, tf.constant(-1, tf.int64))
        return self._decoded_dense

    def build_decoder(self, graph):
        if self._decoder is None:
            if self.config['ctc'] == "greedy":
                self._decoder, _ = tf.nn.ctc_greedy_decoder(
                    graph['logits'], graph['l'], merge_repeated=True)
            elif self.config['ctc']:
                self._decoder, _ = tf.nn.ctc_beam_search_decoder(
                    graph['logits'], graph['l'], merge_repeated=True)
        return self._decoder

    def build_pred_thresholding(self, graph):
        if self._pred_thresholded is None:
            self._pred_thresholded = tf.to_int32(
                graph['class_pred'] > self.config.default('accuracy_threshold', 0.5))
        return self._pred_thresholded

    def build_cer(self, graph):
        if self._cer is None:
            decoded = self.build_decoder(graph)
            self._cer = tf.edit_distance(
                tf.cast(decoded[0], tf.int32), tf.cast(graph['y'], tf.int32))
        return self._cer

    def build_accuracy(self, graph):
        if self._accuracy is None:
            predictions = self.build_pred_thresholding(graph)
            equality = tf.equal(predictions, tf.cast(
                graph['class_y'], tf.int32))
            self._accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        return self._accuracy

    def build_tp(self, graph):
        if self._tp is None:
            pred_res = self.build_pred_res(graph)
            y_res = self.build_y_res(graph)
            self._tp = tf.reduce_mean(
                tf.cast(tf.equal(tf.boolean_mask(pred_res, tf.equal(y_res, 0)), 0), tf.float32))
        return self._tp

    def build_fp(self, graph):
        if self._fp is None:
            pred_res = self.build_pred_res(graph)
            y_res = self.build_y_res(graph)
            self._fp = tf.reduce_mean(
                tf.cast(tf.equal(tf.boolean_mask(pred_res, tf.equal(y_res, 1)), 0), tf.float32))
        return self._fp

    def build_fn(self, graph):
        if self._fn is None:
            pred_res = self.build_pred_res(graph)
            y_res = self.build_y_res(graph)
            self._fn = tf.reduce_mean(
                tf.cast(tf.equal(tf.boolean_mask(pred_res, tf.equal(y_res, 0)), 1), tf.float32))
        return self._fn

    def build_tn(self, graph):
        if self._tn is None:
            pred_res = self.build_pred_res(graph)
            y_res = self.build_y_res(graph)
            self._tn = tf.reduce_mean(
                tf.cast(tf.equal(tf.boolean_mask(pred_res, tf.equal(y_res, 1)), 1), tf.float32))
        return self._tn

    def build_pred_res(self, graph):
        if self._pred_res is None:
            self._pred_res = tf.argmax(graph['output'], 3)
        return self._pred_res

    def build_y_res(self, graph):
        if self._y_res is None:
            _y = tf.cast(tf.reshape(graph['y']/tf.constant(255.0),
                                    [-1, graph['y'].shape[1], graph['y'].shape[2]]), tf.int32)
            _y = tf.one_hot(_y, 2)
            self._y_res = tf.argmax(_y, 3)
        return self._y_res
