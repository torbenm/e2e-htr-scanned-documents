from config.config import Configuration
from .Constants import CONFIG_PATH, MODELS_PATH
from .Transcriber import Transcriber
from .Logger import Logger
from data import util, Dataset, PreparedDataset
from nn import getAlgorithm
from .Executor import Executor


class QuickExecutor(object):

    DEFAULT_CONFIG = {
        "algo_config": {},
        "data_config": {},
    }

    executables = []
    logger = Logger()

    def __init__(self, dataset, configName: str, verbose=False):
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

        if verbose:
            self.config('Algorithm Configuration')
            self.dataset.info()

    def restore(self, date, epoch):
        filename = os.path.join(os.path.join(
            MODELS_PATH, '{}-{}'.format(self.config['name'], date)), 'model-{}'.format(epoch))
        self.executor.restore(filename)

    def add_transcriber(self, subset):
        transcriber = Transcriber(
            self.dataset, self.logger, subset, self.config)
        self.executables.append(transcriber)
        return transcriber

    def __call__(self):
        self.executor(self.executables)
