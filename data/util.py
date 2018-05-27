import os
import sys
import time
from urllib import urlretrieve
import argparse
import json


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


def get_data_path(folder_name):
    # DEPRECATED
    cwd = os.getcwd()
    if cwd[-len(BASE_PATH):] != BASE_PATH:
        cwd = os.path.join(cwd, BASE_PATH)
    return os.path.join(cwd, folder_name)


def rmkdir(folder):
    import shutil
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def delete_folder():
    import shutil
    shutil.rmtree(get_data_path())


def download_all(config, name):
    datapath = get_data_path(name)
    for idx in config:
        download_instance(idx, config[idx], datapath)
    print "Done! All data is prepared."


def parse_auth_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u", "--user", help="Username for iam access", type=str)
    parser.add_argument(
        "-p", "--pwd", help="Password for iam access", type=str)
    return parser.parse_args()


def download_instance(idx, config, basepath):
    download_file = os.path.join(basepath, config["file"])
    if not os.path.exists(download_file):
        do_download(idx, config, download_file)
    if config["tgz"]:
        untarfolder = os.path.join(basepath, os.path.splitext(
            os.path.basename(download_file))[0])
        if not os.path.exists(untarfolder):
            do_untar(idx, download_file, untarfolder)


def progressbar(prefix):
    def _progressbar(step, total):
        percent = int(step / float(total) * 100.0)
        sys.stdout.write("\r{:25} [ {:100} ] {} %".format(
            prefix, "|" * percent, percent))
        sys.stdout.flush()
    return _progressbar


def reporthook(count, block_size, total_size):
    global rh_start_time
    if count == 0:
        rh_start_time = time.time()
        return
    duration = time.time() - rh_start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r" + "|" * percent + "." * (100 - percent) + "%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()


def do_download(idx, config, download_file):
    url = config["url"]
    if config["authenticate"]:
        args = parse_auth_arguments()
        if not args.user or not args.pwd:
            print "Please specify user and password to download", idx
            exit()
        url = "http://{}:{}@{}".format(args.user, args.pwd, url)
        print "Downloading", idx
        urlretrieve(url, download_file, reporthook)


def do_untar(idx, file, folder):
    import tarfile
    with tarfile.open(file, "r:gz") as tar:
        print "\nGoing to untar", idx
        os.mkdir(folder)
        tar.extractall(folder)


def getVocab(text):
    vocab = list(set(text))
    vocab.append("")  # ctc blank label
    vocab_size = len(vocab)
    idx_to_vocab = dict(enumerate(vocab))
    vocab_to_idx = dict(zip(idx_to_vocab.values(), idx_to_vocab.keys()))
    return (idx_to_vocab, vocab_to_idx)


def dump(path, name, vocab):
    # DEPRECATED
    with open(os.path.join(path, "{}.json".format(name)), 'w+') as f:
        json.dump(vocab, f)


def dumpJson(path, name, data):
    with open(os.path.join(path, "{}.json".format(name)), 'w+') as f:
        json.dump(data, f)


def load(path, name):
    # DEPRECATED
    with open(os.path.join(path, "{}.json".format(name)), 'r') as f:
        return json.load(f)


def loadJson(path, name):
    with open(os.path.join(path, "{}.json".format(name)), 'r') as f:
        return json.load(f)


def process_greyscale(imagepath, targetpath, threshold=None, width=None, height=None, scale=1.0):
    import cv2
    image = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
    if width is not None and height is not None and image is not None:
        h, w = image.shape
        ws = max(int(w / scale), 1)
        hs = max(int(h / scale), 1)
        image = cv2.resize(image, (ws, hs))
        image = cv2.copyMakeBorder(
            image, 0, height - hs, 0, width - ws, cv2.BORDER_CONSTANT, value=[255])
    if threshold is not None:
        image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(targetpath, image)


def getscalefactor(imagepaths, target_width, target_height):
    import cv2
    sc = 0
    for ipath in imagepaths:
        image = cv2.imread(ipath)
        if image is not None:
            h, w, _ = image.shape
            sc = max(float(h) / target_height, float(w) / target_width, sc)
    return sc
