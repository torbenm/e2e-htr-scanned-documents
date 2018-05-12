import dataset
import util
import os
import cv2
import numpy as np
import tensorflow as tf

FOLDER_NAME = "iam"
BASE_FOLDER = "data"


class IamDataset(dataset.Dataset):

    def __init__(self, binarize, width, height):
        self._binarize = binarize
        self._width = width
        self._height = height
        self._basepath, self._targetpath, self._targetimagepath = get_targetpath(
            binarize, width, height)
        self.preload()

    def preload(self):
        self._vocab = util.load(self._targetpath, "vocab")
        self._vocab_length = len(self._vocab[0])
        self._lines = util.load(self._targetpath, "lines")
        self._maxlength = max(map(lambda x: len(x["text"]), self._lines))

    def parsetext(self, text):
        length = len(text)
        parsed = [self._vocab[1][c] for c in text]
        parsed.extend([self._vocab_length - 1] * (self._maxlength - length))
        return parsed

    def _loaddata(self):
        X = []
        Y = []
        for line in self._lines:
            x, y = self.loadline(line)
            X.append(x)
            Y.append(y)
        self._raw_x = np.asarray(X)
        self._raw_y = np.asarray(Y)

    def loadline(self, line):
        y = np.asarray(self.parsetext(line["text"]))
        x = cv2.imread(os.path.join(self._targetimagepath, line[
                       "name"] + ".png"), cv2.CV_LOAD_IMAGE_GRAYSCALE)
        x = cv2.normalize(x, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        x = np.transpose(x, [1, 0])
        x = np.reshape(x, [self._width, self._height, 1])
        return x, y

    def generateBatch(self, batch_size):
        total_len = len(self._lines)
        num_batches = total_len // batch_size
        for b in range(num_batches - 1):
            x = self._raw_x[b * batch_size:(b + 1) * batch_size]
            y = self._raw_y[b * batch_size:(b + 1) * batch_size]
            yield x, y
        pass

    def generateEpochs(self, batch_size, num_epochs):
        self._loaddata()
        for e in range(num_epochs):
            yield self.generateBatch(batch_size)

download_data = {
    "lines": {
        "file": "lines.tgz",
        "url":  "www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz",
        "tgz": True,
        "authenticate": True
    },
    "ascii": {
        "file": "ascii.tgz",
        "url": "www.fki.inf.unibe.ch/DBs/iamDB/data/ascii/ascii.tgz",
        "tgz": True,
        "authenticate": True
    }
}


def get_image_path(identifier):
    idsplit = identifier.split("-")
    return os.path.join(idsplit[0], "-".join(idsplit[0:2]), identifier + ".png")


def load_ascii_lines(basepath):
    with open(os.path.join(basepath, "ascii/lines.txt"), "r") as lines:
        parsed = []
        fulltext = ""
        while True:
            line = lines.readline().strip()
            if not line:
                break
            if line[0] != "#":
                lsplit = line.split(" ")
                if lsplit[1] != "ok":
                    path = get_image_path(lsplit[0])
                    parsed.append({"path": path, "name": lsplit[
                                  0], "mean_grey": int(lsplit[2]), "text": lsplit[-1]})
                    fulltext = fulltext + lsplit[-1]
        return parsed, fulltext


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare", help="Prepare the data", action="store_true")
    parser.add_argument(
        "--binarize", help="Should the data be binarized", action="store_true", default=False)
    parser.add_argument(
        "--height", help="Height of the images", type=int, default=30)
    parser.add_argument(
        "--width", help="Width of the images", type=int, default=300)
    return parser.parse_args()


def get_targetpath(binarize, width, height):
    basepath = util.get_data_path("iam")
    dname = "iam-{}-{}x{}".format("nb" if not binarize else "b",
                                  width, height)
    basepath = util.get_data_path("iam")
    targetpath = os.path.join(basepath, "final", dname)
    targetimagepath = os.path.join(targetpath, "img")
    return basepath, targetpath, targetimagepath


def prepare_data(args):
    basepath, targetpath, targetimagepath = get_targetpath(
        args.binarize, args.width, args.height)
    util.rmkdir(targetpath)
    os.makedirs(targetimagepath)
    parsed, fulltext = load_ascii_lines(basepath)
    vocab = util.getVocab(fulltext)
    util.dump(targetpath, "vocab", vocab)
    util.dump(targetpath, "lines", parsed)

    for image in parsed:
        util.process_greyscale(os.path.join(basepath, "lines", image["path"]), os.path.join(targetimagepath, image["name"] + ".png"), image[
            "mean_grey"] if args.binarize else None, args.width, args.height)

if __name__ == "__main__":
    args = parse_args()
    if args.prepare:
        prepare_data(args)
    else:
        util.download_all(download_data, "iam")
