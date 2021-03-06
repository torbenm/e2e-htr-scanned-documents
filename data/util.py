import os
import sys
import time
import requests
from urllib.parse import urlparse
import argparse
import json
import numpy as np


def getFullPath(base, folder_name):
    cwd = os.getcwd()
    if cwd[-len(base):] != base:
        cwd = os.path.join(cwd, base)
    return os.path.join(cwd, folder_name)


BASE_PATH = "data"
CONFIG_PATH = getFullPath(BASE_PATH, "config")
RAW_PATH = getFullPath(BASE_PATH, "raw")
OUTPUT_PATH = getFullPath(BASE_PATH, "output")


def printDone(name, hadProgressBar=False):
    line = "\r" if hadProgressBar else ""
    line = line + "{:45} DONE \n".format(name)
    sys.stdout.write(line)
    sys.stdout.flush()


def printPercentage(name):
    def _printPercentage(step, total):
        percent = int(step / float(total) * 100.0)
        sys.stdout.write("\r{:45} {:2} %".format(name, percent))
        sys.stdout.flush()
    return _printPercentage


def rmkdir(folder):
    import shutil
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def delete_folder():
    import shutil
    shutil.rmtree(get_data_path())


def progressbar(prefix):
    def _progressbar(step, total):
        percent = int(step / float(total) * 100.0)
        sys.stdout.write("\r{:25} [ {:100} ] {} %".format(
            prefix, "|" * percent, percent))
        sys.stdout.flush()
    return _progressbar


def getVocab(text):
    vocab = list(set(text))
    vocab.append("")  # ctc blank label
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
    return (idx_to_vocab, vocab_to_idx)


def dumpJson(path, name, data):
    with open(os.path.join(path, "{}.json".format(name)), 'w+') as f:
        json.dump(data, f)


def loadJson(path, name=None):
    filepath = os.path.join(path, "{}.json".format(
        name)) if name is not None else path
    with open(filepath, 'r') as f:
        return json.load(f)


def retrieve(url, target, username='', password=''):
    filename = os.path.basename(urlparse(url).path)

    r = requests.get(url, auth=(username, password))

    if r.status_code == 200:
        with open(target, 'wb') as out:
            for bits in r.iter_content():
                out.write(bits)


def pad(array, reference_shape, offsets=None):
    """
    array: Array to be padded
    reference_shape: tuple of size of ndarray to create
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    will throw a ValueError if offsets is too big and the reference_shape cannot handle the offsets
    """
    offsets = offsets if offsets is not None else [0] * len(reference_shape)
    # Create an array of zeros with the reference shape
    result = np.zeros(reference_shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + array.shape[dim])
                  for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result
