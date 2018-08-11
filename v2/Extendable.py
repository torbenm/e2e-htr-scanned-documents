import tensorflow as tf
from config.config import Configuration


class Extendable(object):

    _decoded_dense = None
    _decoder = None
    _pred_thresholded = None

    def __init__(self, config={}):
        self.config = Configuration(config)

     def build_decoded_dense(self, graph):
        if self._decoded_dense is None:
            decoded = self._decode(graph)
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
