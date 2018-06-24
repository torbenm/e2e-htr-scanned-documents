import tensorflow as tf
import os
from data import util, dataset
from config.config import Configuration
from nn import getAlgorithm
import time
import numpy as np
from tensorflow.python.client import timeline
from nn.util import valueOr

MODELS_PATH = "./models"
CONFIG_PATH = "./config"
TENSORBOARD_PATH = "./tensorboard"

global_step = 0


class Executor(object):

    def __init__(self, configName, useDataset=None, transpose=False):
        self.config = Configuration(util.loadJson(CONFIG_PATH, configName))
        self._transpose = transpose
        self.algorithm = getAlgorithm(
            self.config['algorithm'], self.config.default('algo_config', {}), self._transpose)
        self.dataset = dataset.Dataset(useDataset or self.config[
                                       'dataset'], self._transpose)
        self.sessionConfig = None
        self._decoder = None
        self._cer = None
        self._decoded_dense = None
        self.config('Algorithm Configuration')
        self.dataset.info()
        self.log_name = '{}-{}'.format(self.config['name'],
                                       time.strftime("%Y-%m-%d-%H-%M-%S"))
        self.tensorboard_path = os.path.join(TENSORBOARD_PATH, self.log_name)
        self.models_path = os.path.join(MODELS_PATH, self.log_name)

    def configure(self, device=-1, softplacement=True, logplacement=False, allow_growth=True):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        if device != -1:
            print('Setting cuda visible devices to', device)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

        print("Configuring. Softplacement: ", softplacement,
              "Logplacement:", logplacement, "Allow growth:", allow_growth)
        self.sessionConfig = tf.ConfigProto(
            allow_soft_placement=softplacement, log_device_placement=logplacement)
        self.sessionConfig.gpu_options.allow_growth = allow_growth
        self.device = evaluate_device(device)
        self.algorithm.set_cpu(device == -1)
        return self.sessionConfig

    def transcribe(self, subset, date=None, epoch=0, hooks=None):
        options = {
            "dataset": subset
        }
        return self._exec(self._transcribe, hooks, date, epoch, options)

    def visualize(self, image, date=None, epoch=0, hooks=None):
        options = {
            "image": image
        }
        return self._exec(self._visualize, hooks, date, epoch, options)

    def train(self, date=None, epoch=0, hooks=None, options={}):
        return self._exec(self._train, hooks, date, epoch, options=options)

    def validate(self, date=None, epoch=0, hooks=None, dataset="dev"):
        options = {
            "dataset": dataset
        }
        return self._exec(self._validate, hooks, date, epoch, options)

    def test(self):
        pass

    def _exec(self, callback, hooks, date=None, epoch=0, options={}):
        print("Going to run on", self.device)
        with tf.device(self.device):
            config = self.sessionConfig or self.configure()
            graph = self._build_graph()
            with tf.Session(config=config) as sess:
                if date is None:
                    sess.run(tf.global_variables_initializer())
                else:
                    self._restore(sess, date, epoch)
                return callback(graph, sess, hooks, options)

    def _train(self, graph, sess, hooks, options={}):
        batch_num = self.dataset.getBatchCount(
            self.config['batch'], self.config['max_batches'])
        self._build_cer(graph)
        summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(sess.graph)
        foldername = os.path.join(self.models_path, 'model')
        if self.config.default('save', False) != False:
            saver = tf.train.Saver(max_to_keep=None)
        for idx, epoch in enumerate(self.dataset.generateEpochs(self.config['batch'], self.config['epochs'], max_batches=self.config['max_batches'])):
            self._train_epoch(
                graph, sess, idx, epoch, batch_num, hooks, options, summ, writer)
            if self.config.default('save', False) != False and (idx % self.config['save'] == 0 or idx == self.config['epochs'] - 1):
                saver.save(sess, foldername, global_step=idx)

    def _train_epoch(self, graph, sess, idx, epoch, batch_num, hooks, options, summ, writer):
        training_loss = 0
        steps = 0
        start_time = time.time()
        run_options = None
        run_metadata = None
        if 'timeline' in options and options['timeline'] != '':
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        # Batch loop
        global_step = 0
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
            training_loss_ = [0]
            if global_step % 100 == 0:
                training_loss_, _, s = sess.run(
                    [graph['total_loss'], graph['train_step'], summ], train_dict,
                    run_metadata=run_metadata, options=run_options)
                writer.add_summary(s, global_step=global_step)
            else:
                training_loss_, _ = sess.run(
                    [graph['total_loss'], graph['train_step']], train_dict,
                    run_metadata=run_metadata, options=run_options)
            global_step += 1
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
        loss_total = []
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
            results_, cer_, loss_ = sess.run(
                [results, cer, graph['total_loss']], val_dict)
            cer_total.append(cer_)
            loss_total.append(np.ma.masked_invalid(
                loss_).mean())
            examples['Y'].extend(Y)
            examples['trans'].extend(results_)

            if hooks is not None and 'val_batch' in hooks:
                hooks['val_batch'](steps, total_steps, cer_)
        return {
            'examples': examples,
            'loss': np.mean(loss_total),
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

    def _visualize(self, graph, sess, hooks=None, options={}):
        X = [self.dataset.load_image(options['image'])]
        viz_dict = {
            graph['x']: X,
            graph['l']: [graph['logits'].shape[0]] * len(X)
        }
        activations = sess.run(graph['viz'], viz_dict)
        return activations

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
            tf.summary.scalar('cer', self._cer)
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
