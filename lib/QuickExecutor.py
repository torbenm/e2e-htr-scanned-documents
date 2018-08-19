import os
import time

from data import util, Dataset, PreparedDataset
from nn import getAlgorithm

from .Configuration import Configuration
from .Constants import CONFIG_PATH, MODELS_PATH
from .executables import RecClassRunner, Saver, Visualizer, ClassTrainer, RecognitionTrainer, RecognitionValidator, ClassValidator, ClassRunner
from .Logger import Logger
from .Executor import Executor


class QuickExecutor(object):

    DEFAULT_CONFIG = {
        "algo_config": {},
        "data_config": {},
    }

    executables = []
    logger = Logger()

    def __init__(self, dataset=None, configName: str="", verbose=False):
        self.config = Configuration(util.loadJson(
            CONFIG_PATH, configName), self.DEFAULT_CONFIG)
        self.algorithm = getAlgorithm(
            self.config['algorithm'], self.config['algo_config'])
        if isinstance(dataset, Dataset.Dataset):
            self.dataset = dataset
        else:
            self.dataset = PreparedDataset.PreparedDataset(dataset or self.config[
                'dataset'], False, self.config['data_config'])

        self.algorithm.configure(batch_size=self.config['batch'], learning_rate=self.config[
            'learning_rate'], sequence_length=self.dataset.max_length,
            image_height=self.dataset.meta["height"], image_width=self.dataset.meta[
            "width"], vocab_length=self.dataset.vocab_length, channels=self.dataset.channels,
            class_learning_rate=self.config.default('class_learning_rate', self.config['learning_rate']))

        self.executor = Executor(
            self.algorithm, verbose, self.config)

        self.log_name = '{}-{}'.format(self.config['name'],
                                       time.strftime("%Y-%m-%d-%H-%M-%S"))
        self.models_path = os.path.join(MODELS_PATH, self.log_name)

        if verbose:
            self.config('Algorithm Configuration')
            self.dataset.info()

    def configure(self, **config):
        self.executor.configure(**config)

    def restore(self, date, epoch):
        filename = os.path.join(os.path.join(
            MODELS_PATH, '{}-{}'.format(self.config['name'], date)), 'model-{}'.format(epoch))
        self.executor.restore(filename)

    def add_transcriber(self, **kwargs):
        return self._add(RecClassRunner, **kwargs)

    def add_classifier(self, **kwargs):
        return self._add(ClassRunner, **kwargs)

    def add_visualizer(self, image):
        return self._add(Visualizer, image=image)

    def add_train_classifier(self, **kwargs):
        return self._add(ClassTrainer, **kwargs)

    def add_train_transcriber(self, **kwargs):
        return self._add(RecognitionTrainer, **kwargs)

    def add_transcription_validator(self, **kwargs):
        return self._add(RecognitionValidator, **kwargs)

    def add_class_validator(self, **kwargs):
        return self._add(ClassValidator, **kwargs)

    def add_saver(self, **kwargs):
        return self._add(Saver, foldername=self.models_path, every_epoch=self.config['save'], **kwargs)

    def _add(self, class_, **kwargs):
        kwargs.setdefault('dataset', self.dataset)
        kwargs.setdefault('logger', self.logger)
        kwargs.setdefault('config', self.config)
        obj = class_(**kwargs)
        self.executables.append(obj)
        return obj

    def add_summary(self):
        self.executor.logger = self.logger

    def __call__(self):
        self.executor(self.executables)
