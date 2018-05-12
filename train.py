import tensorflow as tf
import util
import sys
from time import time
from data.iam import IamDataset
from graves2009 import GravesSchmidhuber2009


def train(graph, dataset, num_epochs=10, batch_size=10, save=False):
    sessionConfig = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=sessionConfig) as sess:
        sess.run(tf.global_variables_initializer())
        batch_num = len(dataset._lines) // batch_size
        for idx, epoch in enumerate(dataset.generateEpochs(batch_size, num_epochs)):
            print "Epoch number", idx
            training_loss = 0
            steps = 0
            start_time = time()
            for X, Y in epoch:
                print "step", steps, "of", batch_num
                steps += 1
                feed_dict = {graph['x']: X, graph['y']: util.denseNDArrayToSparseTensor(Y)}
                training_loss_, _ = sess.run(
                    [graph['total_loss'], graph['train_step']], feed_dict)
                training_loss += training_loss_

            print('epoch = {0} | loss = {1:.3f} | time {2:.3f}'.format(str(idx).zfill(3),
                                                                       training_loss / steps,
                                                                       time() - start_time))
        if isinstance(save, str):
            graph['saver'].save(sess, "saves/{}".format(save))


if __name__ == "__main__":
    # TODO: Probably better to use some sort of library for argument handling
    with tf.device(util.evaluate_device(sys.argv[1])):

        batch_size = 64
        width = 300
        height = 30
        channels = 1
        dataset = IamDataset(True, 300, 30)
        algorithm = GravesSchmidhuber2009()
        print dataset._vocab_length
        graph = algorithm.build_graph(
            batch_size=batch_size, sequence_length=dataset._maxlength, image_height=height, image_width=width, vocab_length=dataset._vocab_length, channels=1)
        train(graph, dataset, num_epochs=1, batch_size=batch_size)
