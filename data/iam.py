import dataset
import util
import os
import cv2
import numpy as np
import tensorflow as tf

FOLDER_NAME = "iam"
BASE_FOLDER = "data"


class IamDataset(dataset.Dataset):

    def __init__(self, binarize, width, height, padding=5):
        self._binarize = binarize
        self._width = width
        self._height = height
        self._padding = padding
        self._loaded = False
        self._basepath, self._targetpath, self._targetimagepath = get_targetpath(
            binarize, width, height)
        self.preload()

    def preload(self):
        self._vocab = util.load(self._targetpath, "vocab")
        self._vocab_length = len(self._vocab[0])
        self._lines = util.load(self._targetpath, "lines")
        # we need to add padding as ctc does not like if there is none or too
        # little (< 2)
        self._maxlength = max(
            map(lambda x: len(x["text"]), self._lines)) + self._padding

    def compile(self, text):
        length = len(text)
        parsed = [self._vocab[1][c] for c in text]
        parsed.extend([self._vocab_length - 1] * self._padding)
        parsed.extend([-1] * (self._maxlength - length - self._padding))
        return parsed

    def decompile(self, values):
        def getKey(key):
            try:
                return self._vocab[0][str(c)]
            except KeyError:
                return '='
        return ''.join([getKey(c) for c in values])

    def _loaddata(self):
        if not self._loaded:
            X = []
            Y = []
            L = []
            for line in self._lines:
                x, y, l = self.loadline(line)
                X.append(x)
                Y.append(y)
                L.append(l)
            self._raw_x = 1 - np.asarray(X)
            self._raw_y = np.asarray(Y)
            self._raw_l = np.asarray(L)
            self._loaded = True

    def loadline(self, line):
        l = len(line["text"]) + self._padding
        y = np.asarray(self.compile(line["text"]))
        x = cv2.imread(os.path.join(self._targetimagepath, line[
                       "name"] + ".png"), cv2.IMREAD_GRAYSCALE)
        x = cv2.normalize(x, x, alpha=0, beta=1,
                          norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        x = np.transpose(x, [1, 0])
        x = np.reshape(x, [self._width, self._height, 1])
        return x, y, l

    def generateBatch(self, batch_size, max_batches=0):
        num_batches = self.getBatchCount(batch_size, max_batches)
        for b in range(num_batches - 1):
            x = self._raw_x[b * batch_size:(b + 1) * batch_size]
            y = self._raw_y[b * batch_size:(b + 1) * batch_size]
            l = self._raw_l[b * batch_size:(b + 1) * batch_size]
            yield x, y, l
        pass

    def prepareDataset(self, validation_size=0, test_size=0, shuffle=False):
        self._loaddata()
        length = len(self._raw_x)
        if shuffle:
            perm = np.random.permutation(length)
            self._raw_x = self._raw_x[perm]
            self._raw_y = self._raw_y[perm]
            self._raw_l = self._raw_l[perm]
        val_length = int(length * validation_size)
        test_length = int(length * test_size)
        self._val_x = self._raw_x[0:val_length]
        self._val_y = self._raw_y[0:val_length]
        self._val_l = self._raw_l[0:val_length]
        self._test_x = self._raw_x[val_length:test_length + val_length]
        self._test_y = self._raw_y[val_length:test_length + val_length]
        self._test_l = self._raw_l[val_length:test_length + val_length]
        #self._raw_x = self._raw_x[test_length + val_length:]
        #self._raw_y = self._raw_y[test_length + val_length:]
        #self._raw_l = self._raw_l[test_length + val_length:]

    def getValidationSet(self):
        return self._val_x, self._val_y, self._val_l

    def getBatchCount(self, batch_size, max_batches=0):
        total_len = len(self._raw_x)
        num_batches = total_len // batch_size
        return min(
            num_batches, max_batches) if max_batches > 0 else num_batches

    def generateEpochs(self, batch_size, num_epochs, max_batches=0):
        for e in range(num_epochs):
            yield self.generateBatch(batch_size, max_batches=max_batches)

download_data = {
    "lines": {
        "file": "lines.tgz",
        "url":  "www.fki.inf.unibe.ch/DBs/iamDB/data/lines/lines.tgz",
        "tgz": True,
        "authenticate": True
    },
    "words": {
        "file": "words.tgz",
        "url":  "www.fki.inf.unibe.ch/DBs/iamDB/data/words/words.tgz",
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


def load_ascii_lines(basepath, type):
    with open(os.path.join(basepath, "ascii/{}.txt".format(type)), "r") as lines:
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
        "--type", help="Words vs. Lines", default="lines")
    parser.add_argument(
        "--height", help="Height of the images", type=int, default=30)
    parser.add_argument(
        "--width", help="Width of the images", type=int, default=300)
    parser.add_argument("--user")
    parser.add_argument("--pwd")
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
    parsed, fulltext = load_ascii_lines(basepath, args.type)
    vocab = util.getVocab(fulltext)
    util.dump(targetpath, "vocab", vocab)
    util.dump(targetpath, "lines", parsed)

    for image in parsed:
        util.process_greyscale(os.path.join(basepath, args.type, image["path"]), os.path.join(targetimagepath, image["name"] + ".png"), image[
            "mean_grey"] if args.binarize else None, args.width, args.height)


if __name__ == "__main__":
    args = parse_args()
    if args.prepare:
        prepare_data(args)
    else:
        util.download_all(download_data, "iam")
