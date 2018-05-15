# -*- coding: utf-8 -*-
import tensorflow as tf
import util
import sys
import numpy as np
import argparse
from time import time
from data.iam import IamDataset
from graves2009 import GravesSchmidhuber2009
from puigcerver2017 import Puigcerver2017
from Voigtlaender2016 import VoigtlaenderDoetschNey2016


def batch_hook(epoch, batch, max_batches):
    percent = (float(batch) / max_batches) * 100
    out = 'epoch = {0} [ {2:100} ] {1:02.2f}% '.format(
        str(epoch).zfill(3), percent, "|" * int(percent))
    sys.stdout.write("\r" + out)
    sys.stdout.flush()


def epoch_hook(epoch, loss, time, ler):
    msg = 'epoch = {0} | loss = {1:.3f} | time {2:.3f} | ler {3:.3f}'.format(str(epoch).zfill(3),
                                                                             loss,
                                                                             time, ler)
    sys.stdout.write('\r{:130}\n'.format(msg))
    sys.stdout.flush()


def compare_outputs(dataset, pred, actual):
    pred = dataset.decompile(pred)
    actual = dataset.decompile(actual)
    out = '{:' + str(dataset._maxlength) + '}  {}'
    return out.format(pred, actual)


def train(graph, dataset, num_epochs=10, batch_size=10, val_size=0.2, shuffle=False, test_size=0, save=False, max_batches=0):
    sessionConfig = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=sessionConfig) as sess:
        # Prepare data
        sess.run(tf.global_variables_initializer())
        dataset.prepareDataset(val_size, test_size, shuffle)
        val_x, val_y, val_l = dataset.getValidationSet()
        val_dict = {graph['x']: val_x[:batch_size],
                    graph['l']: [dataset._compiled_max_length] * batch_size}
        batch_num = dataset.getBatchCount(batch_size, max_batches)
        # Training loop
        for idx, epoch in enumerate(dataset.generateEpochs(batch_size, num_epochs, max_batches=max_batches)):
            training_loss = 0
            steps = 0
            start_time = time()
            # Batch loop
            for X, Y, length in epoch:
                batch_hook(idx, steps, batch_num - 1)
                steps += 1
                train_dict = {graph['x']: X, graph['y']: util.denseNDArrayToSparseTensor(Y), graph[
                    'l']: length}
                training_loss_, other = sess.run(
                    [graph['total_loss'], graph['train_step']], train_dict)
                training_loss += np.ma.masked_invalid(
                    training_loss_).mean()
            # Evaluate training
            preds = sess.run(graph['output'], val_dict)
            print preds.shape
            print '\n'.join([compare_outputs(dataset, preds[c], val_y[c]) for c in range(batch_size)])
            epoch_hook(idx, training_loss / steps, time() - start_time, 0)
        if isinstance(save, str):
            graph['saver'].save(sess, "saves/{}".format(save))


if __name__ == "__main__":
    # TODO: Probably better to use some sort of library for argument handling
    parser = argparse.ArgumentParser()

    parser.add_argument('--binarize', help='Whether dataset should be binarized',
                        action='store_true', default=False)
    parser.add_argument('--width', help='Width of image',
                        default=100, type=int)
    parser.add_argument('--height', help='Height of image',
                        default=50, type=int)
    parser.add_argument('--epochs', help='Number of epochs',
                        default=100, type=int)
    parser.add_argument('--batch', help='Batch size', default=1024, type=int)
    parser.add_argument('--learning-rate',
                        help='Learning Rate', default=5, type=float)
    parser.add_argument('--gpu', help='Runs scripts on gpu. Default is cpu.',
                        action='store_true', default=False)

    args = parser.parse_args()

    with tf.device(util.evaluate_device(args.gpu)):

        dataset = IamDataset(args.binarize, args.width, args.height)
        # channels = dataset._channels
        algorithm = GravesSchmidhuber2009()
        # algorithm = Puigcerver2017()
        # algorithm = VoigtlaenderDoetschNey2016()
        graph = algorithm.build_graph(
            batch_size=args.batch, learning_rate=args.learning_rate, sequence_length=dataset._compiled_max_length, image_height=args.height, image_width=args.width, vocab_length=dataset._vocab_length, channels=dataset._channels)
        train(graph, dataset, num_epochs=args.epochs, batch_size=args.batch)
