import os
import cv2
import numpy as np

from lib.Configuration import Configuration
from lib.buildingblocks.TextSeparation import TextSeparation
from lib.buildingblocks.WordSegmentation import WordSegmentation
from lib.buildingblocks.ParagraphSegmentation import ParagraphSegmentation
from lib.buildingblocks.LineSegmentation import LineSegmentation
from lib.buildingblocks.TranscriptionAndClassification import TranscriptionAndClassification
from lib.buildingblocks.visualizer.RegionVisualizer import RegionVisualizer
from lib.buildingblocks.visualizer.SeparatedVisualizer import SeparatedVisualizer
from lib.buildingblocks.evaluate.gtprovider.WordRegionGTProvider import WordRegionGTProvider
from lib.buildingblocks.evaluate.gtprovider.ParagraphRegionGTProvider import ParagraphRegionGTProvider
from lib.buildingblocks.evaluate.gtprovider.LineRegionGTProvider import LineRegionGTProvider
from lib.buildingblocks.OneGramLanguageModel import OneGramLanguageModel
from lib.buildingblocks.evaluate.IoU import IoU
from lib.buildingblocks.evaluate.IoUPixelSum import IoUPixelSum
from lib.buildingblocks.evaluate.IoUCER import IoUCER
from lib.buildingblocks.evaluate.BagOfWords import BagOfWords
from lib.Logger import Logger
from time import time
from itertools import count, takewhile


def frange(start, stop, step):
    return takewhile(lambda x: x < stop, count(start, step))


class E2ERunner(object):

    def __init__(self, config={}, globalConfig={}):
        self.config = Configuration(config)
        self.globalConfig = Configuration(globalConfig)
        self._parse_config()
        self.logger = Logger()
        self.config()

    def _parse_config(self):
        self._parse_blocks(self.config["blocks"])
        self.viz = self._parse_visualizer(self.config.default("viz", None))
        self.gtprov = self._parse_gt(self.config.default("gt", None))
        self.evals = self._parse_evals(self.config.default('eval', []))

    def _parse_blocks(self, blocks):
        self.blocks = [self._parse_block(
            block) for block in blocks if "disabled" not in block or not block["disabled"]]

    def _parse_block(self, block):
        if block["type"] == "TextSeparation":
            return TextSeparation(self.globalConfig, block)
        elif block["type"] == "WordSegmentation":
            return WordSegmentation(block)
        elif block["type"] == "LineSegmentation":
            return LineSegmentation(block)
        elif block["type"] == "ParagraphSegmentation":
            return ParagraphSegmentation(block)
        elif block["type"] == "OneGramLanguageModel":
            return OneGramLanguageModel(block)
        elif block["type"] == "TranscriptionAndClassification":
            return TranscriptionAndClassification(self.globalConfig, block)

    def _parse_evals(self, eval_configs):
        return [self._parse_eval(config) for config in eval_configs]

    def _parse_eval(self, config):
        if config is None:
            return None
        if config["type"] == "IoU":
            return IoU(config)
        elif config["type"] == "IoUPixelSum":
            return IoUPixelSum(config)
        elif config["type"] == "BagOfWords":
            return BagOfWords(config)
        elif config["type"] == "IoUCER":
            return IoUCER(config)

    def _parse_data(self, data_config):
        if isinstance(data_config, list):
            return data_config
        else:
            filenames = list(filter(lambda f: f.endswith(
                data_config.default("ending")) and f.startswith(data_config.default("prefix", "")), os.listdir(data_config["path"])))
            if data_config["limit"] > 0:
                filenames = filenames[:data_config["limit"]]
            return [os.path.join(data_config["path"], filename) for filename in filenames]

    def _parse_visualizer(self, viz_config):
        if viz_config is None:
            return None
        if viz_config["type"] == "RegionVisualizer":
            return RegionVisualizer(viz_config)
        elif viz_config["type"] == "SeparatedVisualizer":
            return SeparatedVisualizer(viz_config)

    def _parse_gt(self, gt_config):
        if gt_config is None:
            return None
        if gt_config["type"] == "WordRegion":
            return WordRegionGTProvider()
        elif gt_config["type"] == "ParagraphRegion":
            return ParagraphRegionGTProvider()
        elif gt_config["type"] == "LineRegion":
            return LineRegionGTProvider()

    def __call__(self, log_prefix="E2E", skip_range_evaluation=False):
        if not skip_range_evaluation and self.config.default("ranger", False):
            self.logger.write("Entering Range Execution Mode")
            return self._range_exec()
        start = time()
        self.scores = {}
        data = self._parse_data(self.config["data"])
        results = []
        for idx, file in enumerate(data):
            self.logger.progress(log_prefix, idx, len(data))
            results.append(self._exec(file))
        [block.close() for block in self.blocks]
        if len(self.evals) > 0:
            final_scores = {
                "time": time() - start
            }
            for score_key in self.scores:
                final_scores[score_key] = np.average(self.scores[score_key])
            self.logger.summary(log_prefix, final_scores)
        return results

    def _get_range(self):
        if type(self.config["ranger.values"]) is dict:
            return frange(self.config["ranger.values.from"], self.config["ranger.values.to"], self.config["ranger.values.step"])

    def _range_exec(self):
        def set_config(value):
            for path in self.config.default("ranger.paths", [self.config.default("ranger.path", [])]):
                current = self.config
                for step in path[:-1]:
                    current = current[step]
                current[path[-1]] = value
            self._parse_config()

        for val in self._get_range():
            set_config(val)
            prefix = self.config.default("ranger.template", "value {}")
            self(log_prefix=prefix.format(val), skip_range_evaluation=True)

    def _exec(self, file):
        original = cv2.imread(file)
        last_output = original.copy()

        for block in self.blocks:
            last_output = block(last_output)
        res = {
            "file": file,
            "original": original,
            "result": last_output
        }
        if self.gtprov is not None:
            gt = self.gtprov(file, original)
        if self.viz is not None:
            vizimage = res["original"].copy()
            if self.gtprov is not None and self.config.default('gt.viz', False):
                vizimage = self.viz(vizimage, gt, True)
            if len(self.blocks) > 0:
                vizimage = self.viz(vizimage, res["result"], False)
            self.viz.store(vizimage, file)
            res["viz"] = vizimage
        if len(self.evals) > 0:
            for evl in self.evals:
                scores = evl(gt, res["result"])
                for score_key in scores.keys():
                    self.scores[score_key] = [scores[score_key]] if score_key not in self.scores else [
                        scores[score_key], *self.scores[score_key]]
        return res


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    args = parser.parse_args()
    config = Configuration.load("./config/e2e/", args.config)
    e2e = E2ERunner(config, {
        "gpu": args.gpu
    })
    e2e()
