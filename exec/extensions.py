import tensorflow as tf


def decoder(graph, get_ext, config):
    _decoder = None
    if config['ctc'] == "greedy":
        _decoder, _ = tf.nn.ctc_greedy_decoder(
            graph['logits'], graph['l'], merge_repeated=True)
    elif self.config['ctc']:
        _decoder, _ = tf.nn.ctc_beam_search_decoder(
            graph['logits'], graph['l'], merge_repeated=True)
    return _decoder


def character_error_rate(graph, get_ext, config):
    decoded = get_ext('decoder', decoder)
    cer = tf.reduce_mean(tf.edit_distance(
        tf.cast(decoded[0], tf.int32), tf.cast(graph['y'], tf.int32)))
    tf.summary.scalar('cer', self._cer)
    return cer
