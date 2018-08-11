import tensorflow as tf
import os
from data import util, Dataset, PreparedDataset
from config.config import Configuration
from nn import getAlgorithm
import time
import numpy as np
from tensorflow.python.client import timeline
from nn.util import valueOr

MODELS_PATH = "./models"
CONFIG_PATH = "./config"
TENSORBOARD_PATH = "./tensorboard"

global_step = 1
class_step = 1


class Executor(object):

    def __init__(self, configName, _dataset=None, transpose=False, verbose=True):
        self.sessionConfig = None
        self._decoder = None
        self._cer = None
        self._accuracy = None
        self._pred_thresholded = None
        self._decoded_dense = None
        self._graph = None
        self._is_init = False
        self._sess = None
        self.verbose = verbose
        self.log_name = '{}-{}'.format(self.config['name'],
                                       time.strftime("%Y-%m-%d-%H-%M-%S"))
        self.tensorboard_path = os.path.join(TENSORBOARD_PATH, self.log_name)
        self.models_path = os.path.join(MODELS_PATH, self.log_name)
        if verbose:
            self.config('Algorithm Configuration')
            self.dataset.info()

    def _train_denoising(self, graph, sess, hooks, options={}):
        os.makedirs(self.models_path, exist_ok=True)
        if self.config.default('save', False) != False:
            saver = tf.train.Saver(max_to_keep=None)
        files = os.listdir(pdfpath)
        for n in range(self.config['epochs']):
            for filename in files:
                print(filename)

    def _train(self, graph, sess, hooks, options={}):
        batch_num = self.dataset.getBatchCount(
            self.config['batch'], self.config['max_batches'])
        batch_num_printed = 0
        if self.dataset.meta.default('printed', False):
            batch_num_printed = self.dataset.getBatchCount(
                self.config['batch'], self.config['max_batches'], dataset='print_train')
        self._build_cer(graph)
        summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(sess.graph)
        foldername = os.path.join(self.models_path, 'model')
        # Copy configurations to models folder
        os.makedirs(self.models_path, exist_ok=True)
        self.config.save(self.models_path, 'algorithm')
        self.dataset.meta.save(self.models_path, 'data_meta')
        util.dumpJson(self.models_path, 'vocab', self.dataset.vocab)
        self.config.save(self.models_path, 'algorithm')

        class_epochs = None
        if self.dataset.meta.default('printed', False):
            class_epochs = self.dataset.generateEpochs(
                self.config['batch'], self.config['epochs'], max_batches=self.config['max_batches'], dataset='print_train', augmentable=True)
        for idx, epoch in enumerate(self.dataset.generateEpochs(self.config['batch'], self.config['epochs'], max_batches=self.config['max_batches'], augmentable=True)):
            self._train_epoch(
                graph, sess, idx, epoch, batch_num, hooks, options, summ, writer)
            if self.dataset.meta.default('printed', False) and (idx+1) % 3 == 0:
                self._train_class_epoch(graph, sess, idx, next(
                    class_epochs), batch_num_printed, hooks, options, summ, writer)

    def _train_class_epoch(self, graph, sess, idx, epoch, batch_num, hooks, options, summ, writer):
        global class_step
        start_time = time.time()
        training_loss = 0
        steps = 0
        for X, Y, _ in epoch:
            if hooks is not None and 'batch' in hooks:
                hooks['batch'](idx, steps, batch_num)
            steps += 1
            train_dict = {
                graph['x']: X,
                graph['class_y']: Y,
                graph['is_train']: True
            }
            training_loss_ = [0]
            training_loss_, _ = sess.run(
                [graph['class_loss'], graph['class_train']], train_dict)
            class_step += 1
            training_loss += np.ma.masked_invalid(
                training_loss_).mean()
        val_stats = self._validate_classifier(graph, sess, hooks)
        if hooks is not None and 'class_epoch' in hooks:
            hooks['class_epoch'](idx, training_loss / steps,
                                 time.time() - start_time, val_stats)

    def _train_epoch(self, graph, sess, idx, epoch, batch_num, hooks, options, summ, writer):
        global global_step
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
                graph['l']: [self.dataset.max_length] * len(X),
                graph['is_train']: True
            }
            training_loss_ = [0]
            if global_step % 50 == 0:
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
            cer_total.extend(cer_)
            loss_total.extend(loss_)
            examples['Y'].extend(Y)
            examples['trans'].extend(results_)

            if hooks is not None and 'val_batch' in hooks:
                hooks['val_batch'](steps, total_steps, cer_)
        return {
            'examples': examples,
            'loss': np.ma.masked_invalid(loss_total).mean(),
            'cer': np.mean(cer_total)
        }

    def _validate_classifier(self, graph, sess, hooks=None, options={}):
        # OPTIONS
        dataset = options['dataset'] if 'dataset' in options else 'print_dev'

        # ADDITIONAL GRAPHs
        accuracy = self._build_accuracy(graph)
        # VARIABLES
        steps = 0
        total_steps = self.dataset.getBatchCount(
            self.config['batch'], self.config['max_batches'], dataset)
        acc_total = []
        for X, Y, _ in self.dataset.generateBatch(self.config['batch'], self.config['max_batches'], dataset):
            steps += 1
            val_dict = {
                graph['x']: X,
                graph['class_y']: Y
            }
            acc_ = sess.run(
                accuracy, val_dict)
            acc_total.append(acc_)
            if hooks is not None and 'val_batch' in hooks:
                hooks['val_batch'](steps, total_steps, acc_)
        return {
            "accuracy": np.mean(acc_total)
        }

    def _empty_val_stats(self):
        return {
            'examples': {
                'Y': [],
                'trans': []
            },
            'loss': 0,
            'cer': -1.0,
            'accuracy': -1.0
        }

    def _investigate(self, graph, sess, hooks=None, options={}):
        # OPTIONS
        dataset = options['dataset'] if 'dataset' in options else 'test'

        # ADDITIONAL GRAPHs
        results = self._build_decoded_dense(graph)
        class_pred = self._build_pred_thresholding(graph)
        cer = self._build_cer(graph)

        # VARIABLES
        steps = 0
        total_steps = self.dataset.getBatchCount(
            self.config['batch'], self.config['max_batches'], dataset)

        transcriptions = {
            'files': [],
            'trans': [],
            'class': [],
            'cer': [],
            'truth': []
        }
        for X, Y, L, F in self.dataset.generateBatch(self.config['batch'], self.config['max_batches'], dataset, True):
            steps += 1
            val_dict = {
                graph['x']: X,
                graph['y']: denseNDArrayToSparseTensor(Y),
                graph['l']: [self.dataset.max_length] * len(X)
                #graph['l']: [graph['logits'].shape[0]] * len(X)
            }
            if self.dataset.meta.default('printed', False):
                results_, cer_, class_ = sess.run(
                    [results, cer, graph['class_pred']], val_dict)
                transcriptions['class'].extend(class_)
            else:
                results_, cer_ = sess.run([results, cer], val_dict)
            transcriptions['files'].extend(F)
            transcriptions['cer'].extend(cer_)
            transcriptions['truth'].extend(Y)
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

    def _build_cer(self, graph):
        if self._cer is None:
            decoded = self._decode(graph)
            self._cer = tf.edit_distance(
                tf.cast(decoded[0], tf.int32), tf.cast(graph['y'], tf.int32))
            tf.summary.scalar('cer', tf.reduce_mean(self._cer))
        return self._cer

    def _build_accuracy(self, graph):
        if self._accuracy is None:
            predictions = self._build_pred_thresholding(graph)
            equality = tf.equal(predictions, tf.cast(
                graph['class_y'], tf.int32))
            self._accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
        return self._accuracy

    def _restore(self, sess, date="", epoch=0):
        filename = os.path.join(self.get_model_path(
            date), 'model-{}'.format(epoch))
        tf.train.Saver().restore(sess, filename)

    def get_model_path(self, date):
        return os.path.join(
            MODELS_PATH, '{}-{}'.format(self.config['name'], date))


def denseNDArrayToSparseTensor(arr, sparse_val=-1):
    idx = np.where(arr != sparse_val)
    return tf.SparseTensorValue(np.vstack(idx).T, arr[idx], arr.shape)
