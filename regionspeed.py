import argparse
import cv2
import numpy as np
from time import time
# from segmentation.MSERRegionExtractor import RegionExtractor
from segmentation.WordRegionExtractor import RegionExtractor
from data.RegionDataset import RegionDataset
from data.util import loadJson, rmkdir
from eval.evaluate import evaluate
import executor
import os
import re

# "htrnet-pc-iam-print"
# otf-iam-both-2018-08-07-15-38-49
ALGORITHM_CONFIG = "otf-iam-paper"
# "2018-07-07-14-59-06"  # "2018-07-02-23-46-51"
MODEL_DATE = "2018-08-22-22-10-32"
# 800  # 65
MODEL_EPOCH = 24

DATAPATH = "../paper-notes/data/final"
SUBSET = "dev"


PUNCTUATION_REGEX = re.compile(r"([|])(?=[,.;:!?])")
REGULAR_REGEX = re.compile(r"[|]")

HTR_THRESHOLD = 0.8


class RegionExecutor(object):
    def _regions(self, img):
        return RegionExtractor(img).extract()

    def __call__(self, imgpath: str):
        img = cv2.imread(imgpath)
        regions = self._regions(img)

    def paper_notes(self, basepath, num):
        return self(os.path.join(basepath, "{}-paper.png".format(num)))


if __name__ == "__main__":

    def get_num(name: str):
        return name.split(".")[0].split("-")[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--limit', default=-1, type=int)

    parser.add_argument('--datapath', default=DATAPATH)
    parser.add_argument('--subset', default=SUBSET)
    args = parser.parse_args()

    basepath = os.path.join(args.datapath, args.subset)

    exc = RegionExecutor()

    files = os.listdir(basepath)
    files = list(map(lambda num: "{}-truth.json".format(num), filenums))

    idx = 0
    scores = []
    start = time()
    mx = min(len(files), args.limit) if args.limit != -1 else len(files)
    for file in files:
        print("{} of {}".format(idx, mx))
        if idx >= args.limit and not args.limit == -1:
            break
        if file.endswith("json"):
            num = get_num(file)
            exc.paper_notes(basepath, num)
            idx += 1
    total_time = time() - start
    print("Took {:.2f}s".format(total_time))
    print("That's {:.2f}s per image".format(total_time/idx))
