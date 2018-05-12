# -*- coding: utf-8 -*-
import tensorflow as tf
import util
import sys
import numpy as np
from time import time
from data.iam import IamDataset
from graves2009 import GravesSchmidhuber2009


def batch_hook(epoch, batch, max_batches):
    percent = (float(batch) / max_batches) * 100
    out = u'epoch = {0} [ {2:100} ] {1:02.2f}% '.format(
        str(epoch).zfill(3), percent, u"\u2588" * int(percent))
    sys.stdout.write(u"\r" + out)
    sys.stdout.flush()


def epoch_hook(epoch, loss, time):
    msg = 'epoch = {0} | loss = {1:.3f} | time {2:.3f}'.format(str(epoch).zfill(3),
                                                               loss,
                                                               time)
    sys.stdout.write('\r{:130}\n'.format(msg))
    sys.stdout.flush()


def train(graph, dataset, num_epochs=10, batch_size=10, save=False, max_batches=0):
    sessionConfig = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=sessionConfig) as sess:
        sess.run(tf.global_variables_initializer())
        batch_num = len(dataset._lines) // batch_size
        for idx, epoch in enumerate(dataset.generateEpochs(batch_size, num_epochs, max_batches=max_batches)):
            training_loss = 0
            steps = 0
            start_time = time()
            for X, Y, length in epoch:
                batch_hook(idx, steps, batch_num - 1)
                steps += 1
                feed_dict = {graph['x']: X, graph['y']: util.denseNDArrayToSparseTensor(Y)}
                training_loss_, other = sess.run(
                    [graph['total_loss'], graph['train_step']], feed_dict)
                training_loss += np.ma.masked_invalid(
                    training_loss_).mean()
            epoch_hook(idx, training_loss / steps, time() - start_time)
        if isinstance(save, str):
            graph['saver'].save(sess, "saves/{}".format(save))


if __name__ == "__main__":
    # TODO: Probably better to use some sort of library for argument handling
    with tf.device(util.evaluate_device(sys.argv[1])):

        batch_size = 128
        num_epochs = 10
        width = 300
        height = 30
        channels = 1
        dataset = IamDataset(True, 300, 30)
        algorithm = GravesSchmidhuber2009()
        graph = algorithm.build_graph(
            batch_size=batch_size, sequence_length=dataset._maxlength, image_height=height, image_width=width, vocab_length=dataset._vocab_length, channels=1)
        train(graph, dataset, num_epochs=num_epochs, batch_size=batch_size)
