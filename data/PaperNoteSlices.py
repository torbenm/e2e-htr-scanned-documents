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


class PaperNoteSlices(Dataset):
    # just need to be regenerated if the dataset changes completely!
    average_sizes = {
        "dev": [3078, 2217],
        "test": [3066, 2206],
        "train": [3079, 2225]
    }

    def __init__(self, **kwargs):
        self.paper_note_path = kwargs.get(
            'paper_note_path', '../paper-notes/data/final')
        self.slice_width = kwargs.get('slice_width', 320)
        self.slice_height = kwargs.get('slice_height', 320)
        self.meta = Configuration({})
        self.vocab = {}
        self._load_filelists()

    def info(self):
        pass

    def _load_filelists(self):
        self.data = {
            "dev": self._load_filelist("dev"),
            "train": self._load_filelist("train"),
            "test": self._load_filelist("test")
        }

    def _load_filelist(self, subset):
        basepath = os.path.join(self.paper_note_path, subset)
        all_files = os.listdir(basepath)
        files = list(filter(lambda x: x.endswith("-paper.png"), all_files))
        return list(map(lambda x: self._fileobj(basepath, x), files))

    def _fileobj(self, basepath: str, filename: str):
        num = filename.split("-")[0]
        return {
            "paper": os.path.join(basepath, filename),
            "stripped": os.path.join(basepath, "{}-stripped.png".format(num)),
        }

    def _slice(self, img):
        slices = []
        img = np.reshape(img, [img.shape[0], img.shape[1], 1])
        for w_i in range(int(img.shape[1]/self.slice_width)):
            for h_i in range(int(img.shape[0]/self.slice_height)):
                slc = img[(h_i*self.slice_height):((h_i+1)*self.slice_height),
                          (w_i*self.slice_width):((w_i+1)*self.slice_width), :]
                slices.append(slc)
        return slices

    def compile(self, text):
        return text

    def decompile(self, values):
        return values

    def _load_file(self, fileobj):
        paper = cv2.imread(fileobj["paper"], cv2.IMREAD_GRAYSCALE)
        stripped = cv2.imread(fileobj["stripped"], cv2.IMREAD_GRAYSCALE)
        _, stripped = cv2.threshold(stripped, 250, 255, cv2.THRESH_BINARY)
        # cv2.imshow('stripped', cv2.resize(stripped, (600, 800)))
        # cv2.waitKey(0)
        if self.slice_height == -1 and self.slice_width == -1:
            paper = np.reshape(paper, [paper.shape[0], paper.shape[1], 1])
            stripped = np.reshape(
                stripped, [stripped.shape[0], stripped.shape[1], 1])
            return [paper], [stripped]
        slices_paper, slices_stripped = self._slice(
            paper), self._slice(stripped)
        final_paper, final_stripped = [], []
        for i in range(len(slices_paper)):
            m = np.min(slices_stripped[i])
            if m < 125:
                final_paper.append(slices_paper[i])
                final_stripped.append(slices_stripped[i])
        return final_paper, final_stripped

    def _get_slices(self, paper, stripped, free):
        if free > len(paper):
            return paper, stripped, [], []
        else:
            return paper[:free], stripped[:free], paper[free:], stripped[free:]

    def _process_labels(self, label):
        label = np.asarray(label)
        label = np.int32(label/255.0)
        nx = label.shape[1]
        ny = label.shape[0]
        label = np.reshape(label, (ny, nx))
        labels = np.zeros((ny, nx, 2), dtype=np.float32)
        labels[..., 1] = label
        labels[..., 0] = 1 - label
        return labels

    # override
    def generateBatch(self, batch_size=0, max_batches=0, dataset="train", with_filepath=False):
        files = self.data[dataset]
        shuffle(files)
        surplus_paper = []
        surplus_stripped = []
        batch_paper = []
        batch_stripped = []
        total_batches = 0
        batch_size = batch_size if self.slice_height != - \
            1 or self.slice_width != -1 else 1
        for file in files:
            if self.slice_height == -1 and self.slice_width == -1:
                batch_paper, batch_stripped = self._load_file(file)
            else:
                if len(surplus_paper) > 0 and len(batch_paper) < batch_size:
                    new_paper, new_stripped, surplus_paper, surplus_stripped = self._get_slices(
                        surplus_paper, surplus_stripped, batch_size - len(batch_paper))
                    batch_paper.extend(new_paper)
                    batch_stripped.extend(new_stripped)

                if len(batch_paper) < batch_size:
                    paper, stripped = self._load_file(file)
                    new_paper, new_stripped, surplus_paper, surplus_stripped = self._get_slices(
                        paper, stripped, batch_size - len(batch_paper))
                    batch_paper.extend(new_paper)
                    batch_stripped.extend(new_stripped)

            if len(batch_paper) >= batch_size:
                batch_paper = np.asarray(batch_paper)/255.0
                Y_ = []
                for y in batch_stripped:
                    Y_.append(self._process_labels(y))
                if with_filepath:
                    yield batch_paper, Y_, [], []
                else:
                    yield batch_paper, Y_, []
                total_batches += 1
                if max_batches > 0 and total_batches >= max_batches:
                    break
                batch_paper = []
                batch_stripped = []
        pass

    # override
    def generateEpochs(self, batch_size, num_epochs, max_batches=0, dataset="train", with_filepath=False):
        return [self.generateBatch()]

    # override
    def getBatchCount(self, batch_size, max_batches=0, dataset="train"):
        batch_size = batch_size if self.slice_height != - \
            1 or self.slice_width != -1 else 1
        if self.slice_height == -1 and self.slice_width == -1:
            num_batches = len(self.data[dataset])
        else:
            num_batches = np.floor(float(np.prod(self.average_sizes[dataset]))/np.prod(
                [self.slice_height, self.slice_width]))*len(self.data[dataset])
        batch_count = np.ceil(num_batches/batch_size)
        return batch_count if max_batches == 0 else min(max_batches, batch_count)

    def _averageSize(self, subset):
        def get_image_size(file):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            return img.shape[:2]
        sizes = list(map(lambda x: get_image_size(
            x["paper"]), self.data[subset]))
        return np.average(sizes, axis=0)


if __name__ == "__main__":
    pns = PaperNoteSlices(paper_note_path='../paper-notes/data/final')
    for X, Y, _ in pns.generateBatch(50):
        for x in X:
            print(x.shape)
            cv2.imshow('x', x)
            cv2.waitKey(0)
    # pns.generateBatch()
