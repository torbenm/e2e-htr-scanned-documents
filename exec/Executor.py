import tensorflow as tf
import os
from config.config import Configuration
from nn import getAlgorithm
from data import util, dataset
import time
import numpy as np


from tensorflow.python.client import timeline
from nn.util import valueOr

MODELS_PATH = "./models"
CONFIG_PATH = "./config"
TENSORBOARD_PATH = "./tensorboard"

    self.config('Algorithm Configuration')
    self.dataset.info()
    self.log_name = '{}-{}'.format(self.config['name'],
                                   time.strftime("%Y-%m-%d-%H-%M-%S"))
    self.tensorboard_path = os.path.join(TENSORBOARD_PATH, self.log_name)
    self.models_path = os.path.join(MODELS_PATH, self.log_name)


class Executor(object):

    def __init__(self, configFile, legacy={}, indiv_dataset=None):
        self._legacy = Configuration(legacy)

        # Load Dataset, Algorithm, Configuration...
        self.config = Configuration(util.loadJson(CONFIG_PATH, configName))
        self.algorithm = getAlgorithm(
            self.config['algorithm'], self.config.default('algo_config', {}), self.is_legacy('transpose'))
        self.dataset = dataset.Dataset(indiv_dataset or self.config[
            'dataset'], self.is_legacy('transpose'))

        # Initialize variables
        self._extensions = {}  # graph extensions
        self.log_name = '{}-{}'.format(self.config['name'],
                                       time.strftime("%Y-%m-%d-%H-%M-%S"))
        self._paths = {
            "tensorboard": os.path.join(TENSORBOARD_PATH, self.log_name),
            "models": os.path.join(MODELS_PATH, self.log_name)
        }
        self._restore_model = None

        # Display information
        self.config('Algorithm Configuration')
        self.dataset.info()

    def is_legacy(self, prop):
        return self._legacy.default(prop, False)

    def extend_graph(self, name, extension):
        if name not in self._extensions:
            self._extensions['name'] = extension(self.graph, self.config)
        return self._extensions[name]

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

    def restore(self, model):
        self._restore_model = model

    # Execute
    def __call__(self):
        pass

    def _build_graph(self):
        return self.algorithm.build_graph(
            batch_size=self.config['batch'],
            learning_rate=self.config['learning_rate'],
            sequence_length=self.dataset.max_length,
            image_height=self.dataset.meta["height"],
            image_width=self.dataset.meta["width"],
            vocab_length=self.dataset.vocab_length,
            channels=self.dataset.channels,
            class_learning_rate=self.config.default('class_learning_rate', self.config['learning_rate']))

    # Execution helper
    def _run_in_context(self, exec_fn):
        """
        exec_fn(sess) should take exactly one argument: the session passed
        """
        print("Executing on device", self.device)
        with tf.device(self.device):
            config = self.sessionConfig or self.configure()
            self.graph = self._build_graph()
            with tf.Session(config=config) as sess:
                if not self._restore_model:
                    sess.run(tf.global_variables_initializer())
                else:
                    self._run_restoration(sess)
                return exec_fn(sess)

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
        if self.config.default('save', False) != False:
            saver = tf.train.Saver(max_to_keep=None)
        class_epochs = None
        if self.dataset.meta.default('printed', False):
            class_epochs = self.dataset.generateEpochs(
                self.config['batch'], self.config['epochs'], max_batches=self.config['max_batches'], dataset='print_train')
        for idx, epoch in enumerate(self.dataset.generateEpochs(self.config['batch'], self.config['epochs'], max_batches=self.config['max_batches'])):
            self._train_epoch(
                graph, sess, idx, epoch, batch_num, hooks, options, summ, writer)
            if self.dataset.meta.default('printed', False):
                self._train_class_epoch(graph, sess, idx, next(
                    class_epochs), batch_num, hooks, options, summ, writer)
            if self.config.default('save', False) != False and (idx % self.config['save'] == 0 or idx == self.config['epochs'] - 1):
                saver.save(sess, foldername, global_step=idx)

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
                graph['l']: length,
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

    def _run_epoch(self, sess, epoch_data, post_batch):

    def _run_restoration(self, sess):
        tf.train.Saver().restore(sess, self._restore_model)

    def _evaluate_device(self, gpuNumber):
    return "/device:CPU:0" if gpuNumber == -1 else "/device:GPU:{}".format(gpuNumber)

    def _denseNDArrayToSparseTensor(self, arr, sparse_val=-1):
        idx = np.where(arr != sparse_val)
        return tf.SparseTensorValue(np.vstack(idx).T, arr[idx], arr.shape)