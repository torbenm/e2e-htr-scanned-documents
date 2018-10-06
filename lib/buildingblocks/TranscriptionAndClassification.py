from lib.Executor import Executor
from lib.Configuration import Configuration
from nn.htrnet import HtrNet

DEFAULTS = {
    "classify": False,
    "model_path": "",
    "model_epich": 0,
}

GLOBAL_DEFAULTS = {
    "hardplacement": False,
    "logplacement": False,
    "gpu": -1,
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
        self.executor = Executor() QuickExecutor(
            dataset=None, configName="", verbose=False)
        self.executor.configure(softplacement=not self["hardplacement"],
                                logplacement=self["logplacement"], device=self.globalConfig["gpu"])
        self.transcriber = self.executor.add_transcriber("")
        self.executor.restore(args.model_date, args.model_epoch)  # TODO

    def _configure_algorithm(self):
        self.algorithm = HtrNet(self.modelConfig["algo_config"])
        self.algorithm.set_cpu(self.config['device'] == -1)
        self.algorithm.configure(batch_size=self.algoConfig['batch'], learning_rate=self.algoConfig[
            'learning_rate'], sequence_length=self.dataset.max_length,
            image_height=self.dataset.meta["height"], image_width=self.dataset.meta[
            "width"], vocab_length=self.dataset.vocab_length, channels=self.dataset.channels,
            class_learning_rate=self.algoConfig.default('class_learning_rate', self.algoConfig['learning_rate']))

    def _configure_dataset(self):
        pass

    def _configure_executor(self):
        pass

    def __call__(self, images):
        pass

    def close(self):
        pass
