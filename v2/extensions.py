import tensorflow as tf


def decoder(logits, length, mode="greedy"):
    _decoder = None
    if mode == "greedy":
        _decoder, _ = tf.nn.ctc_greedy_decoder(
            logits, length, merge_repeated=True)
    elif mode:
        _decoder, _ = tf.nn.ctc_beam_search_decoder(
            logits, length, merge_repeated=True)
    return _decoder


def character_error_rate(graph, get_ext, config):
    decoded = get_ext('decoder', decoder)
    cer = tf.reduce_mean(tf.edit_distance(
        tf.cast(decoded[0], tf.int32), tf.cast(graph['y'], tf.int32)))
    tf.summary.scalar('cer', self._cer)
    return cer
