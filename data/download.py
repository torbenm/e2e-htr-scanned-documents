from datasets import identify
import util
import os
from urllib import urlretrieve
import tarfile


def process_instance(name, config, path, args):
    download_file = os.path.join(path, config["file"])
    if not os.path.exists(download_file):
        do_download(name, config, download_file, args)
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
            print "Please specify user and password to download", name
            exit()
        url = "{}://{}:{}@{}".format(config['protocol'],
                                     args.user, args.pwd, url)
    urlretrieve(url, download_target, reporthook(name))
    util.printDone("Downloading {}".format(name), True)


def do_untar(name, file, folder):
    with tarfile.open(file, "r:gz") as tar:
        os.mkdir(folder)
        tar.extractall(folder)
        util.printDone("Untarring {}".format(name))


def downloadDataset(dataset, args):
    # STEP 0: Get Download Info
    download_info = dataset.getDownloadInfo()
    raw_path = os.path.join(util.RAW_PATH, dataset.getIdentifier())

    # STEP 1: Make Download Path
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
    util.printDone('Setting up folders')

    # STEP 2: Iterate through files
    for identifier in download_info:
        process_instance(identifier, download_info[identifier], raw_path, args)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", help="Dataset to download", default="iam")
    parser.add_argument("--user", default="")
    parser.add_argument("--pwd", default="")
    args = parser.parse_args()

    dataset = identify.identifyDataset(args.dataset)
    downloadDataset(dataset, args)
