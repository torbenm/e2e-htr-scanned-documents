import os
import cv2
import numpy as np

from lib.Configuration import Configuration
from lib.buildingblocks.TextSeparation import TextSeparation
from lib.buildingblocks.WordSegmentation import WordSegmentation
from lib.buildingblocks.LineSegmentation import LineSegmentation
from lib.buildingblocks.TranscriptionAndClassification import TranscriptionAndClassification
from lib.buildingblocks.visualizer.RegionVisualizer import RegionVisualizer

EXAMPLE_CONFIG = {
    "blocks": [
        # {
        #     "type": "TextSeparation",
        #     "model_path": "models/separation-2018-09-19-17-04-46",
        #     "model_epoch": 207
        # },
        # {
        #     "type": "LineSegmentation"
        # },
        {
            "type": "WordSegmentation"
        },
        {
            "type": "TranscriptionAndClassification",
            "classify": True,
            "model_path": "models/otf-iam-paper-2018-08-28-23-10-33",
            "model_epoch": 74
        }
        # ... building blocks
    ],
    "eval": [
        # ... evaluators
    ],
    "viz": {
        "type": "RegionVisualizer"
    },
    "data": {
        # array of files or object with path + ending
        "path": "../paper-notes/data/final/dev/",
        "ending": "-stripped.png",
        "limit": 1
    }
}


class E2ERunner(object):

    def __init__(self, config={}, globalConfig={}):
        self.config = Configuration(config)
        self.globalConfig = Configuration(globalConfig)
        self._parse_blocks(self.config["blocks"])
        self.viz = self._parse_visualizer(self.config.default("viz", None))

    def _parse_blocks(self, blocks):
        self.blocks = [self._parse_block(block) for block in blocks]

    def _parse_block(self, block):
        if block["type"] == "TextSeparation":
            return TextSeparation(self.globalConfig, block)
        elif block["type"] == "WordSegmentation":
            return WordSegmentation(block)
        elif block["type"] == "LineSegmentation":
            return LineSegmentation(block)
        elif block["type"] == "TranscriptionAndClassification":
            return TranscriptionAndClassification(self.globalConfig, block)

    def _parse_data(self, data_config):
        if isinstance(data_config, list):
            return data_config
        else:
            filenames = list(filter(lambda f: f.endswith(
                data_config["ending"]), os.listdir(data_config["path"])))
            if data_config["limit"] > 0:
                filenames = filenames[:data_config["limit"]]
            return [os.path.join(data_config["path"], filename) for filename in filenames]

    def _parse_visualizer(self, viz_config):
        if viz_config is None:
            return None
        if viz_config["type"] == "RegionVisualizer":
            return RegionVisualizer(viz_config)

    def __call__(self):
        vizzed = []
        results = [self._exec(file)
                   for file in self._parse_data(self.config["data"])]
        [block.close() for block in self.blocks]
        if self.viz is not None:
            vizzed = [self._viz(result) for result in results]
        return results, vizzed

    def _exec(self, file):
        original = cv2.imread(file)
        last_output = original.copy()
        for block in self.blocks:
            last_output = block(last_output)
        return {
            "file": file,
            "original": original,
            "result": last_output
        }

    def _viz(self, res):
        return self.viz(res["original"], res["result"])


if __name__ == "__main__":
    e2e = E2ERunner(EXAMPLE_CONFIG)
    res, viz = e2e()
    cv2.imwrite("e2e_out.png", viz[0])
