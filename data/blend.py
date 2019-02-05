import cv2
import numpy as np
from lib.Configuration import Configuration
from data.datasets import identifyDataset
from data.steps.blender import PageHandwritingBlender
from . import util
import os
import sys
import random

CONFIG = Configuration({
    "data": {
        "line": "iam",
        "line_subset": "lines",
        "pages": "paper/png"
    },
    "lines_per_page": 7,
    "blending": {}
})


pdfpath = os.path.join(util.RAW_PATH, CONFIG['data.pages'])
basepath = os.path.join(util.OUTPUT_PATH, "blended")
util.rmkdir(basepath)
truthpath = os.path.join(basepath, "truth")
imgpath = os.path.join(basepath, "img")
os.makedirs(truthpath)
os.makedirs(imgpath)

# 1. Load Lines
line_dataset = identifyDataset(CONFIG['data.line'])
lines, _ = line_dataset.getFilesAndTruth(
    util.RAW_PATH, CONFIG['data.line_subset'])

# 2. Iterate over pngs of pdfs at randomly select lines to attach
files = os.listdir(pdfpath)
i = 0
for filename in files:
    i += 1
    line = "\r{:150}".format(
        "Processing {} of {} ({})...".format(i, len(files), filename))
    sys.stdout.write(line)
    sys.stdout.flush()
    if filename.endswith(".png"):
        filepath = os.path.join(pdfpath, filename)
        paper = cv2.imread(filepath)
        phb = PageHandwritingBlender(paper, CONFIG['blending'])
        for _ in range(CONFIG['lines_per_page']):
            linepath = random.choice(lines)['path']
            line = cv2.imread(linepath)
            phb(line)
        phb.save(os.path.join(imgpath, filename),
                 os.path.join(truthpath, filename))
