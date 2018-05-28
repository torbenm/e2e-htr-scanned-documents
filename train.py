# -*- coding: utf-8 -*-
import tensorflow as tf
import util
import sys
import numpy as np
import argparse
from time import time
from data.dataset import Dataset


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


def train(graph, dataset, num_epochs=10, batch_size=10, val_size=1, shuffle=False, test_size=0, save='', max_batches=0, softplacement=True, logplacement=False):
    sessionConfig = tf.ConfigProto(
        allow_soft_placement=softplacement, log_device_placement=logplacement)
    sessionConfig.gpu_options.allow_growth = True
    with tf.Session(config=sessionConfig) as sess:
        # Prepare data
        sess.run(tf.global_variables_initializer())
        val_x, val_y, val_l = dataset._load_batch(0, batch_size, "dev")
        val_dict = {graph['x']: val_x[:batch_size],
                    graph['l']: [dataset.max_length] * batch_size}
        batch_num = dataset.getBatchCount(batch_size, max_batches)
        # Training loop
        for idx, epoch in enumerate(dataset.generateEpochs(batch_size, num_epochs, max_batches=max_batches)):
            training_loss = 0
            steps = 0
            start_time = time()
            # Batch loop
            for X, Y, length in epoch:
                batch_hook(idx, steps, batch_num)
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
            print '\n'.join([util.compare_outputs(dataset, preds[c],
                                                  val_y[c]) for c in range(batch_size)])
            epoch_hook(idx, training_loss / steps, time() - start_time, 0)
        if save != '':
            graph['saver'].save(sess, "saves/{}".format(save))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', help='Number of epochs',
                        default=100, type=int)
    parser.add_argument('--batch', help='Batch size', default=1024, type=int)
    parser.add_argument('--learning-rate',
                        help='Learning Rate', default=0.0005, type=float)
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument('--softplacement', help='Allow Softplacement, default is True',
                        action='store_true', default=False)
    parser.add_argument('--logplacement', help='Log Device placement',
                        action='store_true', default=False)
    parser.add_argument('--save', help='save', default='')
    parser.add_argument('--dataset', help='dataset', default='iam-lines')
    parser.add_argument('--algorithm', help='Algorithm', default='puigcerver')

    args = parser.parse_args()

    with tf.device(util.evaluate_device(args.gpu)):

        dataset = Dataset(args.dataset)
        algorithm = util.getAlgorithm(args.algorithm)
        graph = algorithm.build_graph(
            batch_size=args.batch, learning_rate=args.learning_rate, sequence_length=dataset.max_length,
            image_height=dataset.meta["height"], image_width=dataset.meta["width"], vocab_length=dataset.vocab_length, channels=dataset.channels)

        train(graph, dataset, num_epochs=args.epochs, save=args.save,
              batch_size=args.batch, softplacement=args.softplacement, logplacement=args.logplacement)
