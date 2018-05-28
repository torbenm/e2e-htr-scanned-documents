import tensorflow as tf
import os
from data import util, dataset
from nn import getAlgorithm
import time
import numpy as np


MODELS_PATH = "./models"
CONFIG_PATH = "./config"


class Executor(object):

    def __init__(self, configName):
        self.config = util.loadJson(CONFIG_PATH, configName)
        self.algorithm = getAlgorithm(self.config['algorithm'])
        self.dataset = dataset.Dataset(self.config['dataset'])
        self.sessionConfig = None

    def configure(self, device=-1, softplacement=True, logplacement=False, allow_growth=True):
        self.sessionConfig = tf.ConfigProto(
            allow_soft_placement=softplacement, log_device_placement=logplacement)
        self.sessionConfig.gpu_options.allow_growth = allow_growth
        self.device = evaluate_device(device)
        return self.sessionConfig

    def forward(self, hooks):
        pass

    def train(self, hooks=None):
        return self._exec(self._train, hooks)

    def validate(self, hooks=None):
        return self._exec(self._validate, hooks)

    def test(self):
        pass

    def _exec(self, callback, hooks):
        graph = self._build_graph()
        config = self.sessionConfig or self.configure()
        with tf.device(self.device):
            with tf.Session(config=config) as sess:
                return callback(graph, sess, hooks)

    def _train(self, graph, sess, hooks):
        sess.run(tf.global_variables_initializer())
        batch_num = self.dataset.getBatchCount(
            self.config['batch'], self.config['max_batches'])
        for idx, epoch in enumerate(self.dataset.generateEpochs(self.config['batch'], self.config['epochs'], max_batches=self.config['max_batches'])):
            self._train_epoch(graph, sess, idx, epoch, batch_num, hooks)
        if 'save' in self.config and self.config['save']:
            tf.train.Saver().save(sess, os.path.join(
                MODELS_PATH, '{}-{}'.format(self.config['name'], time.strftime("%Y-%m-%d-%H-%M-%S")), 'model'))

    def _train_epoch(self, graph, sess, idx, epoch, batch_num, hooks):
        training_loss = 0
        steps = 0
        start_time = time.time()
        # Batch loop
        for X, Y, length in epoch:
            if hooks is not None and 'batch' in hooks:
                hooks['batch'](idx, steps, batch_num)
            steps += 1
            train_dict = {
                graph['x']: X,
                graph['y']: denseNDArrayToSparseTensor(Y),
                graph['l']: length,
                graph['is_train']: True
            }
            training_loss_, other = sess.run(
                [graph['total_loss'], graph['train_step']], train_dict)
            training_loss += np.ma.masked_invalid(
                training_loss_).mean()
        val_stats = self._validate(graph, sess)
        if hooks is not None and 'epoch' in hooks:
            hooks['epoch'](idx, training_loss / steps,
                           time.time() - start_time, val_stats)

    def _validate(self, graph, sess, hooks=None):
        ler_total = []

        decoded = self._decode(graph)

        ler = tf.reduce_mean(tf.edit_distance(
            tf.cast(decoded[0], tf.int32), tf.cast(graph['y'], tf.int32)))

        for X, Y, L in self.dataset.generateBatch(self.config['batch'], self.config['max_batches'], "dev"):
            val_dict = {
                graph['x']: X,
                graph['y']: denseNDArrayToSparseTensor(Y),
                graph['l']: [self.dataset.max_length] * len(X)
            }
            ler = sess.run(ler, val_dict)
            ler_total.append(ler)
        return {
            'ler': np.mean(ler_total)
        }

    def _decode(self, graph):
        decoded = None
        if self.config['ctc'] == "greedy":
            decoded, _ = tf.nn.ctc_greedy_decoder(
                graph['logits'], graph['l'], merge_repeated=True)
        elif self.config['ctc']:
            decoded, _ = tf.nn.ctc_beam_search_decoder(
                graph['logits'], graph['l'], merge_repeated=True)
        return decoded

    def _build_graph(self):
        return self.algorithm.build_graph(
            batch_size=self.config['batch'], learning_rate=self.config[
                'learning_rate'], sequence_length=self.dataset.max_length,
            image_height=self.dataset.meta["height"], image_width=self.dataset.meta["width"], vocab_length=self.dataset.vocab_length, channels=self.dataset.channels)


def evaluate_device(gpuNumber):
    return "/device:CPU:0" if gpuNumber == -1 else "/device:GPU:{}".format(gpuNumber)


def denseNDArrayToSparseTensor(arr, sparse_val=-1):
    idx = np.where(arr != sparse_val)
    return tf.SparseTensorValue(np.vstack(idx).T, arr[idx], arr.shape)
