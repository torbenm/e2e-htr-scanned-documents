import os
import numpy as np

from lib.Executor import Executor
from lib.Configuration import Configuration
from lib.executables.SeparationRunner import SeparationRunner
from nn.tfunet import TFUnet
from data.PaperNoteSlicesSingle import PaperNoteSlicesSingle

DEFAULTS = {
    "model_path": "",
    "model_epoch": 0,
    "subset": "",
    "binarize_method": ""
}

GLOBAL_DEFAULTS = {
    "hardplacement": False,
    "logplacement": False,
    "gpu": -1,
    "allowGrowth": True
}


class TextSeparation(object):

    name = "Text Separation"

    def __init__(self, globalConfig={}, config={}):
        self.globalConfig = Configuration(globalConfig, GLOBAL_DEFAULTS)
        self.config = Configuration(config, DEFAULTS)
        self.modelConfig = Configuration.load(
            self.config["model_path"], "algorithm")
        self._configure_dataset()
        self._configure_algorithm()
        self._configure_executor()

    def _configure_algorithm(self):
        self.algorithm = TFUnet(self.modelConfig["algo_config"])
        self.algorithm.set_cpu(self.globalConfig['gpu'] == -1)
        self.algorithm.configure(
            slice_width=self.modelConfig['data_config.slice_width'], slice_height=self.modelConfig['data_config.slice_height'])

    def _configure_dataset(self):
        self.dataset = PaperNoteSlicesSingle(
            slice_width=self.modelConfig['data_config.slice_width'],
            slice_height=self.modelConfig['data_config.slice_height'],
            binarize=self.modelConfig.default("binary", False),
            binarize_method=self.config["binarize_method"]
        )

    def _configure_executor(self):
        self.executor = Executor(self.algorithm, False, self.globalConfig)
        self.executor.configure(softplacement=not self.globalConfig["hardplacement"],
                                logplacement=self.globalConfig["logplacement"], device=self.globalConfig["gpu"])
        self.executor.restore(os.path.join(
            self.config["model_path"], "model-{}".format(self.config["model_epoch"])))
        self.separator = SeparationRunner(config=self.modelConfig,
                                          dataset=self.dataset, subset=self.config["subset"])
        self.executables = [self.separator]

    def __call__(self, image):
        original = self.dataset.set_image(image).copy()
        self.executor(self.executables,  auto_close=False)
        outputs = np.argmax(np.asarray(self.separator.outputs), 3)
        merged = self.dataset.merge_slices(outputs, original.shape)
        return (255-(1-merged)*(255-original))

    def close(self):
        self.executor.close()
