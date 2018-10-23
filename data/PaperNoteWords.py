from . import util
from .Dataset import Dataset
from lib.Configuration import Configuration
import os
import math
import numpy as np
import cv2
import sys
from random import shuffle
from data.steps.pipes import warp, morph, convert
from .Dataset import Dataset
from data.steps.pipes import crop, threshold, invert, padding
from data.ImageAugmenter import ImageAugmenter


class PaperNoteWords(Dataset):
    def __init__(self, **kwargs):
        self.paper_note_path = kwargs.get(
            'paper_note_path', '../paper-notes/data/words')
        self.meta = Configuration(kwargs.get('meta', {}))
        self.data_config = Configuration(kwargs.get('data_config', {}))
        self.vocab = kwargs.get('vocab', {})
        self.pure = kwargs.get('pure', True)

        self.max_length = kwargs.get('max_length')
        self._load_data()
        self._compile_sets()
        self.augmenter = ImageAugmenter(self.data_config)

    def info(self):
        pass

    def _compile_set(self, dataset):
        for item in self.data[dataset]:
            item['compiled'] = self.compile(item['truth'])

    def _compile_sets(self):
        self._compile_set("train")
        self._compile_set("dev")
        self._compile_set("test")

    def _load_data(self):
        prefix = "pure_" if self.pure else ""
        self.data = {
            "dev": self._load_wordlist("{}dev".format(prefix)),
            "train": self._load_wordlist("{}train".format(prefix)),
            "test": self._load_wordlist("{}test".format(prefix)),
            "print_dev": self._load_classlist("dev"),
            "print_test": self._load_classlist("test"),
            "print_train": self._load_classlist("train"),
        }

    def _load_wordlist(self, subset):
        basepath = os.path.join(self.paper_note_path, subset)
        words = util.loadJson(basepath, "words")
        parsed = []
        for word in words:
            parsed.append(self._fileobj(
                basepath, "{}.png".format(word), words[word]))
        return parsed

    def _load_classlist(self, subset):
        files = self._load_filelist(subset, 1)
        files.extend(self._load_filelist(
            "print_{}".format(subset), 0, len(files)))
        return files

    def _load_filelist(self, subset, is_htr, length=None) -> list:
        basepath = os.path.join(self.paper_note_path, subset)
        if os.path.exists(basepath):
            all_files = os.listdir(basepath)
            shuffle(all_files)
            length = len(all_files) if length is None else min(
                length, len(all_files))
            files = list(
                filter(lambda x: x.endswith(".png"), all_files[:length]))
            return list(map(lambda x: self._fileobj(basepath, x, is_htr), files))
        return []

    def _fileobj(self, basepath: str, filename: str, truth):
        return {
            "path": os.path.join(basepath, filename),
            "truth": truth,
        }

    def compile(self, text):
        parsed = [self.vocab[1][c] for c in text]
        parsed.extend([-1] * (self.max_length - len(text)))
        return parsed

    def decompile(self, values):
        def getKey(key):
            try:
                return self.vocab[0][str(key)]
            except KeyError:
                return ''
        return ''.join([getKey(c) for c in values])

    def getBatchCount(self, batch_size, max_batches=0, dataset="train"):
        total_len = len(self.data[dataset])
        num_batches = int(math.ceil(float(total_len) / batch_size))
        return min(
            num_batches, max_batches) if max_batches > 0 else num_batches

    def generateBatch(self, batch_size, max_batches=0, dataset="train", with_filepath=False, augmentable=False):
        num_batches = self.getBatchCount(batch_size, max_batches, dataset)
        if self.data_config.default('shuffle_epoch', False):
            shuffle(self.data[dataset])
        for b in range(num_batches):
            yield self._load_batch(b, batch_size, dataset, with_filepath, augmentable=augmentable)
        pass

    def load_image(self, path, transpose=False, augmentable=False):
        target_size = (
            int(self.meta["height"] -
                (self.data_config.default('preprocess.padding', 0)*2)),
            int(self.meta["width"] -
                (self.data_config.default('preprocess.padding', 0)*2))
        )
        x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if x is None or x.shape[0] == 0 or x.shape[1] == 0:
            return None
        x = self.augmenter.preprocess(x, target_size)
        if x is None:
            return None
        if self.data_config.default("otf_augmentations", False) and augmentable:
            x = self.augmenter.augment(x)
        else:
            x = self.augmenter.add_graychannel(x)

        if x.shape[1] != self.meta["width"] or x.shape[0] != self.meta["height"]:
            x = self.augmenter.pad_to_size(
                x, width=self.meta["width"], height=self.meta["height"])

        return self.augmenter.add_graychannel(x)

    def _loadline(self, line, transpose=True, augmentable=False):
        l = len(line["truth"])
        y = np.asarray(line["compiled"])
        x = self.load_image(line["path"], augmentable=augmentable)
        return x, y, l, line["path"]

    def _loadprintline(self, line, transpose=True, augmentable=False):
        y = line["truth"]
        x = self.load_image(line["path"], augmentable=augmentable)
        return x, [y], 0, line["path"]

    def _load_batch(self, index, batch_size, dataset, with_filepath=False, augmentable=False):
        X = []
        Y = []
        L = []
        F = []

        parseline = self._loadline if not dataset.startswith(
            "print_") else self._loadprintline

        for idx in range(index * batch_size, min((index + 1) * batch_size, len(self.data[dataset]))):
            x, y, l, f = parseline(
                self.data[dataset][idx], augmentable=augmentable)
            if x is not None:
                X.append(x)
                Y.append(y)
                L.append(l)
                F.append(f)
        X = np.asarray(X)
        Y = np.asarray(Y)
        L = np.asarray(L)
        if not with_filepath:
            return X, Y, L
        else:
            return X, Y, L, F

    # deprecated

    def generateEpochs(self, batch_size, num_epochs, max_batches=0, dataset="train", with_filepath=False, augmentable=False):
        for e in range(num_epochs):
            yield self.generateBatch(batch_size, max_batches=max_batches, dataset=dataset, with_filepath=with_filepath, augmentable=augmentable)
