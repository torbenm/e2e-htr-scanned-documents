from lib.Executor import Executor
from lib.Configuration import Configuration
from lib.executables.RecClassRunner import RecClassRunner
from nn.htrnet import HtrNet
from data.RegionDataset import RegionDataset

DEFAULTS = {
    "classify": False,
    "model_path": "",
    "model_epoch": 0,
    "scaling": 2.15,
    "max_height": 123,
    "max_width": 1049,
    "max_batches": 0
}

GLOBAL_DEFAULTS = {
    "hardplacement": False,
    "logplacement": False,
    "gpu": -1,
    "allowGrowth": True
}


class TranscriptionAndClassification(object):

    def __init__(self, globalConfig={}, config={}):
        self.globalConfig = Configuration(globalConfig, GLOBAL_DEFAULTS)
        self.config = Configuration(config, DEFAULTS)
        self.modelConfig = Configuration.load(
            self.config["model_path"], "algorithm")

        ################
        # EXECUTOR
        #################
        self.executor = Executor(alg)

    def _configure_algorithm(self):
        self.algorithm = HtrNet(self.modelConfig["algo_config"])
        self.algorithm.set_cpu(self.config['device'] == -1)
        self.algorithm.configure(batch_size=self.algoConfig['batch'], learning_rate=self.algoConfig[
            'learning_rate'], sequence_length=self.dataset.max_length,
            image_height=self.dataset.meta["height"], image_width=self.dataset.meta[
            "width"], vocab_length=self.dataset.vocab_length, channels=self.dataset.channels,
            class_learning_rate=self.algoConfig.default('class_learning_rate', self.algoConfig['learning_rate']))

    def _configure_dataset(self):
        self.dataset = RegionDataset(None,  self.config["model_path"])
        self.dataset.scaling(
            self.config["scaling"], self.config["max_height"], self.config["max_width"])

    def _configure_executor(self):
        self.executor = Executor(self.algorithm, False, self.globalConfig)
        self.executor.configure(softplacement=not self.globalConfig["hardplacement"],
                                logplacement=self.globalConfig["logplacement"], device=self.globalConfig["gpu"])
        self.executor.restore(os.path.join(
            self.config["model_path"], "model-{}".format(self.config["model_epoch"])))
        self.transcriber = RecClassRunner(self.dataset, config=self.config)
        self.executables = [self.transcriber]

    def __call__(self, images):
        self.dataset.set_regions(images)
        self.executor(self.executables, auto_close=False)
        # TODO: returning...

    def close(self):
        self.executor.close()
