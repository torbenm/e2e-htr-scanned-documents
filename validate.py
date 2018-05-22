import tensorflow as tf
import util
import sys
import numpy as np
import argparse
from time import time
from data.iam import IamDataset


def validate(graph, data, model, decoder, batch_size, softplacement, logplacement, val_size=1, shuffle=False, test_size=0):
    sessionConfig = tf.ConfigProto(
        allow_soft_placement=softplacement, log_device_placement=logplacement)
    with tf.Session(config=sessionConfig) as sess:
        graph['saver'].restore(sess, "saves/{}".format(model))

        dataset.prepareDataset(val_size, test_size, batch_size, shuffle)

        val_x, val_y, val_l = dataset.getValidationSet()
        val_dict = {graph['x']: val_x[:batch_size],
                    graph['y']: util.denseNDArrayToSparseTensor(val_y[:batch_size]),
                    graph['l']: [dataset.maxLength()] * batch_size}

        if decoder == "greedy":
            decoded, _ = tf.nn.ctc_greedy_decoder(
                graph['logits'], graph['l'], merge_repeated=True)
        elif decoder == "beam":
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                graph['logits'], graph['l'], merge_repeated=True)

        print graph['y'].dense_shape, decoded[0].dense_shape

        ler = tf.reduce_mean(tf.edit_distance(
            tf.cast(decoded[0], tf.int32), graph['y']))

        decoded = tf.sparse_to_dense(
            decoded[0].indices, decoded[0].dense_shape, decoded[0].values)

        preds, ler = sess.run([decoded, ler], val_dict)
        print preds.shape
        print '\n'.join([util.compare_outputs(dataset, preds[c], val_y[c]) for c in range(batch_size)])
        print 'Edit Distance:', ler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--binarize', help='Whether dataset should be binarized',
                        action='store_true', default=False)
    parser.add_argument('--width', help='Width of image',
                        default=100, type=int)
    parser.add_argument('--height', help='Height of image',
                        default=50, type=int)
    parser.add_argument('--batch', help='Batch size',
                        default=1024, type=int)
    parser.add_argument('--learning-rate',
                        help='Learning Rate', default=0.0005, type=float)
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument('--softplacement', help='Allow Softplacement, default is True',
                        action='store_true', default=False)
    parser.add_argument('--logplacement', help='Log Device placement',
                        action='store_true', default=False)
    parser.add_argument('--model', help='Model', default='')
    parser.add_argument('--algorithm', help='Algorithm',
                        default='puigcerver')
    parser.add_argument('--decoder', help='Decoder', default='beam')

    args = parser.parse_args()

    with tf.device(util.evaluate_device(args.gpu)):

        dataset = IamDataset(args.binarize, args.width, args.height)
        algorithm = util.getAlgorithm(args.algorithm)
        graph = algorithm.build_graph(
            batch_size=args.batch, learning_rate=args.learning_rate, sequence_length=dataset.maxLength(),
            image_height=args.height, image_width=args.width, vocab_length=dataset._vocab_length, channels=dataset._channels)
        validate(graph, dataset, model=args.model, decoder=args.decoder,
                 batch_size=args.batch, softplacement=args.softplacement, logplacement=args.logplacement)
