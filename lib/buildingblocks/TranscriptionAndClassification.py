import os

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
    "max_batches": 0,
    "class_thresh": 0.5
}

GLOBAL_DEFAULTS = {
    "hardplacement": False,
    "logplacement": False,
    "gpu": -1,
    "allowGrowth": True
}

# todo: move classify down to recClassrunner to not do unneccessary work


class TranscriptionAndClassification(object):

    def __init__(self, globalConfig={}, config={}):
        self.globalConfig = Configuration(globalConfig, GLOBAL_DEFAULTS)
        self.config = Configuration(config, DEFAULTS)
        self.modelConfig = Configuration.load(
            self.config["model_path"], "algorithm")
        self._configure_dataset()
        self._configure_algorithm()
        self._configure_executor()

    def _configure_algorithm(self):
        self.algorithm = HtrNet(self.modelConfig["algo_config"])
        self.algorithm.set_cpu(self.globalConfig['gpu'] == -1)
        self.algorithm.configure(batch_size=self.modelConfig['batch'], learning_rate=self.modelConfig[
            'learning_rate'], sequence_length=self.dataset.max_length,
            image_height=self.dataset.meta["height"], image_width=self.dataset.meta[
            "width"], vocab_length=self.dataset.vocab_length, channels=self.dataset.channels,
            class_learning_rate=self.modelConfig.default('class_learning_rate', self.modelConfig['learning_rate']))

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
        self.transcriber = RecClassRunner(
            self.dataset, config=self.modelConfig)
        self.executables = [self.transcriber]

    def __call__(self, images):
        self.dataset.set_regions(images)
        self.executor(self.executables, auto_close=False)
        for idx in range(len(self.transcriber.transcriptions['trans'])):
            images[idx].text = self.dataset.decompile(
                self.transcriber.transcriptions['trans'][idx])
            if self.config["classify"]:
                images[idx]._class = self.transcriber.transcriptions['class'][idx] > self.config["class_thresh"]
        return images

    def close(self):
        self.executor.close()
