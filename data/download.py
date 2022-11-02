from .datasets import identifyDataset
from . import util
import os
from urllib.request import urlretrieve
import tarfile


def process_instance(name, config, path, args):
    download_file = os.path.join(path, config["file"])
    if not os.path.exists(download_file):
        if 'url' in config:
            do_download(name, config, download_file, args)
        else:
            print(config['error'])
            exit()
    if "tgz" in config and config["tgz"]:
        untarfolder = os.path.join(path, os.path.splitext(
            os.path.basename(download_file))[0])
        if not os.path.exists(untarfolder):
            do_untar(name, download_file, untarfolder)


def reporthook(name):
    innerhook = util.printPercentage("Downloading {}".format(name))

    def _innerhook(count, block_size, total_size):
        innerhook(count * block_size, total_size)

    return _innerhook


def do_download(name, config, download_target, args):
    url = config["url"]
    if config["authenticate"]:
        if not args.user or not args.pwd:
            print("Please specify user and password to download", name)
            exit()
        url = "{}://{}".format(config['protocol'],
                               url)
    util.retrieve(url, download_target, args.user, args.pwd)
    util.printDone("Downloading {}".format(name), True)


def do_untar(name, file, folder):
    with tarfile.open(file, "r:*") as tar:
        os.mkdir(folder)
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, folder)
        util.printDone("Untarring {}".format(name))


def downloadDataset(dataset, args):
    # STEP 0: Get Download Info
    raw_path = os.path.join(util.RAW_PATH, dataset.getIdentifier())
    download_info = dataset.getDownloadInfo()

    # STEP 1: Make Download Path
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    util.printDone('Setting up folders')

    # STEP 2: Iterate through files
    for identifier in download_info:
        process_instance(identifier, download_info[identifier], raw_path, args)

    if dataset.requires_splitting:
        dataset.do_split(raw_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", help="Dataset to download", default="iam")
    parser.add_argument("--user", default="")
    parser.add_argument("--pwd", default="")
    args = parser.parse_args()

    dataset = identifyDataset(args.dataset)
    downloadDataset(dataset, args)
