import tensorflow as tf
import os
from data import util, dataset
from nn import getAlgorithm
import time
import numpy as np
from tensorflow.python.client import timeline


MODELS_PATH = "./models"
CONFIG_PATH = "./config"


class Executor(object):

    def __init__(self, configName, useDataset=None):
        self.config = util.loadJson(CONFIG_PATH, configName)
        self.algorithm = getAlgorithm(self.config['algorithm'])
        self.dataset = dataset.Dataset(useDataset or self.config['dataset'])
        self.sessionConfig = None
        self._decoder = None
        self._cer = None
        self._decoded_dense = None

    def configure(self, device=-1, softplacement=False, logplacement=False, allow_growth=True):
        print "Configuring. Softplacement: ", softplacement, "Logplacement:", logplacement, "Allow growth:", allow_growth
        self.sessionConfig = tf.ConfigProto(
            allow_soft_placement=softplacement, log_device_placement=logplacement)
        self.sessionConfig.gpu_options.allow_growth = allow_growth
        self.device = evaluate_device(device)
        return self.sessionConfig

    def transcribe(self, subset, date=None, epoch=0, hooks=None):
        options = {
            "dataset": subset
        }
        return self._exec(self._transcribe, hooks, date, epoch, options)

    def train(self, hooks=None, options={}):
        return self._exec(self._train, hooks, options=options)

    def validate(self, date=None, epoch=0, hooks=None, dataset="dev"):
        options = {
            "dataset": dataset
        }
        return self._exec(self._validate, hooks, date, epoch, options)

    def test(self):
        pass

    def _exec(self, callback, hooks, date=None, epoch=0, options={}):
        graph = self._build_graph()
        config = self.sessionConfig or self.configure()
        with tf.device(self.device):
            with tf.Session(config=config) as sess:
                if date is None:
                    sess.run(tf.global_variables_initializer())
                else:
                    self._restore(sess, date, epoch)
                return callback(graph, sess, hooks, options)

    def _train(self, graph, sess, hooks, options={}):
        batch_num = self.dataset.getBatchCount(
            self.config['batch'], self.config['max_batches'])
        foldername = os.path.join(
            MODELS_PATH, '{}-{}'.format(self.config['name'], time.strftime("%Y-%m-%d-%H-%M-%S")), 'model')
        if 'save' in self.config and self.config['save'] != False:
            saver = tf.train.Saver()
        for idx, epoch in enumerate(self.dataset.generateEpochs(self.config['batch'], self.config['epochs'], max_batches=self.config['max_batches'])):
            self._train_epoch(
                graph, sess, idx, epoch, batch_num, hooks, options)
            if 'save' in self.config and self.config['save'] != False and (idx % self.config['save'] == 0 or idx == self.config['epochs'] - 1):
                saver.save(sess, foldername, global_step=idx)

    def _train_epoch(self, graph, sess, idx, epoch, batch_num, hooks, options):
        training_loss = 0
        steps = 0
        start_time = time.time()
        run_options = None
        run_metadata = None
        if 'timeline' in options and options['timeline'] != '':
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
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
                [graph['total_loss'], graph['train_step']], train_dict,
                run_metadata=run_metadata, options=run_options)

            training_loss += np.ma.masked_invalid(
                training_loss_).mean()
        if 'skip_validation' in options and options['skip_validation']:
            val_stats = self._empty_val_stats()
        else:
            val_stats = self._validate(graph, sess, hooks)
        if hooks is not None and 'epoch' in hooks:
            hooks['epoch'](idx, training_loss / steps,
                           time.time() - start_time, val_stats)
        if 'timeline' in options and options['timeline'] != '':
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timelines/%s_epoch_%d.json' % (options['timeline'], idx), 'w') as f:
                f.write(chrome_trace)

    def _validate(self, graph, sess, hooks=None, options={}):
        # OPTIONS
        dataset = options['dataset'] if 'dataset' in options else 'dev'

        # ADDITIONAL GRAPHs
        cer = self._build_cer(graph)
        results = self._build_decoded_dense(graph)
        # VARIABLES
        steps = 0
        total_steps = self.dataset.getBatchCount(
            self.config['batch'], self.config['max_batches'], dataset)
        cer_total = []
        examples = {
            'Y': [],
            'trans': []
        }
        for X, Y, L in self.dataset.generateBatch(self.config['batch'], self.config['max_batches'], dataset):
            steps += 1
            val_dict = {
                graph['x']: X,
                graph['y']: denseNDArrayToSparseTensor(Y),
                graph['l']: [self.dataset.max_length] * len(X)
            }
            results_, cer_ = sess.run([results, cer], val_dict)
            cer_total.append(cer_)
            examples['Y'].extend(Y)
            examples['trans'].extend(results_)

            if hooks is not None and 'val_batch' in hooks:
                hooks['val_batch'](steps, total_steps, cer_)
        return {
            'examples': examples,
            'cer': np.mean(cer_total)
        }

    def _empty_val_stats(self):
        return {
            'examples': {
                'Y': [],
                'trans': []
            },
            'cer': -1.0
        }

    def _transcribe(self, graph, sess, hooks=None, options={}):
        # OPTIONS
        dataset = options['dataset'] if 'dataset' in options else 'test'

        # ADDITIONAL GRAPHs
        results = self._build_decoded_dense(graph)
        # VARIABLES
        steps = 0
        total_steps = self.dataset.getBatchCount(
            self.config['batch'], self.config['max_batches'], dataset)

        transcriptions = {
            'files': [],
            'trans': []
        }
        for X, Y, L, F in self.dataset.generateBatch(self.config['batch'], self.config['max_batches'], dataset, True):
            steps += 1
            val_dict = {
                graph['x']: X,
                graph['l']: [graph['logits'].shape[0]] * len(X)
            }
            results_ = sess.run(results, val_dict)
            transcriptions['files'].extend(F)
            transcriptions['trans'].extend(results_)

            if hooks is not None and 'trans_batch' in hooks:
                hooks['trans_batch'](steps, total_steps)
        return transcriptions

    def _build_decoded_dense(self, graph):
        if self._decoded_dense is None:
            decoded = self._decode(graph)
            self._decoded_dense = tf.sparse_to_dense(
                decoded[0].indices, decoded[0].dense_shape, decoded[0].values, tf.constant(-1, tf.int64))
        return self._decoded_dense

    def _decode(self, graph):
        if self._decoder is None:
            if self.config['ctc'] == "greedy":
                self._decoder, _ = tf.nn.ctc_greedy_decoder(
                    graph['logits'], graph['l'], merge_repeated=True)
            elif self.config['ctc']:
                self._decoder, _ = tf.nn.ctc_beam_search_decoder(
                    graph['logits'], graph['l'], merge_repeated=True)
        return self._decoder

    def _build_cer(self, graph):
        if self._cer is None:
            decoded = self._decode(graph)
            self._cer = tf.reduce_mean(tf.edit_distance(
                tf.cast(decoded[0], tf.int32), tf.cast(graph['y'], tf.int32)))
        return self._cer

    def _build_graph(self):
        return self.algorithm.build_graph(
            batch_size=self.config['batch'], learning_rate=self.config[
                'learning_rate'], sequence_length=self.dataset.max_length,
            image_height=self.dataset.meta["height"], image_width=self.dataset.meta["width"], vocab_length=self.dataset.vocab_length, channels=self.dataset.channels)

    def _restore(self, sess, date="", epoch=0):
        filename = os.path.join(
            MODELS_PATH, '{}-{}'.format(self.config['name'], date), 'model-{}'.format(epoch))
        tf.train.Saver().restore(sess, filename)


def evaluate_device(gpuNumber):
    return "/device:CPU:0" if gpuNumber == -1 else "/device:GPU:{}".format(gpuNumber)


def denseNDArrayToSparseTensor(arr, sparse_val=-1):
    idx = np.where(arr != sparse_val)
    return tf.SparseTensorValue(np.vstack(idx).T, arr[idx], arr.shape)
