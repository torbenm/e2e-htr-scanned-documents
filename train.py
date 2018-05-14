# -*- coding: utf-8 -*-
import tensorflow as tf
import util
import sys
import numpy as np
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


def epoch_hook(epoch, loss, time):
    msg = 'epoch = {0} | loss = {1:.3f} | time {2:.3f}'.format(str(epoch).zfill(3),
                                                               loss,
                                                               time)
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
                    graph['l']: [dataset._maxlength] * batch_size}
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
            epoch_hook(idx, training_loss / steps, time() - start_time)
        if isinstance(save, str):
            graph['saver'].save(sess, "saves/{}".format(save))


if __name__ == "__main__":
    # TODO: Probably better to use some sort of library for argument handling
    with tf.device(util.evaluate_device(sys.argv[1])):

        batch_size = 256
        num_epochs = 100
        width = 50
        height = 50
        channels = 1
        dataset = IamDataset(False, width, height)
        # channels = dataset._channels
        algorithm = GravesSchmidhuber2009()
        # algorithm = Puigcerver2017()
        # algorithm = VoigtlaenderDoetschNey2016()
        graph = algorithm.build_graph(
            batch_size=batch_size, sequence_length=dataset._maxlength, image_height=height, image_width=width, vocab_length=dataset._vocab_length, channels=1)
        train(graph, dataset, num_epochs=num_epochs, batch_size=batch_size)
