import os
from . import util
from .datasets import identifyDataset
from .steps import vocab, scalefactor, pipeline, split, index, enhance, generate_print
import sys
import numpy as np
from random import shuffle


def loadContext(name):
    return util.loadJson(util.CONFIG_PATH, name)


def prepareDataset(name, context, ehanceDataset):
    dataset = identifyDataset(context['dataset'])
    if dataset is not None:
        basepath = os.path.join(util.OUTPUT_PATH, name)
        imagepath = os.path.join(basepath, "imgs")
        printedpath = os.path.join(basepath, "printed")

        # Step 0: Prepare folders
        util.rmkdir(basepath)
        os.makedirs(imagepath)
        context['imagetarget'] = imagepath
        util.printDone("Setting up folders")

        # Step 1: Parse files & fulltext
        files, fulltext = dataset.getFilesAndTruth(
            util.RAW_PATH, context['subset'], context['limit'] if 'limit' in context else -1)
        util.printDone("Reading raw data")

        # Step 2: Extract & save vocab from fulltext
        if "vocab" in context:
            vocabTuple = util.loadJson(os.path.join(
                util.OUTPUT_PATH, context['vocab']), "vocab")
        else:
            vocabTuple = vocab.getVocab(fulltext)
        util.dumpJson(basepath, "vocab", vocabTuple)
        util.printDone("Extracting vocabulary")

        # Step 3: Extract scale factor, if wished
        if 'scale' in context and ('mode' not in context['scale'] or context['scale']['mode'] == 'factor'):
            imagepaths = list(map(lambda x: x["path"], files))
            context['scale']['factor'] = scalefactor.getScaleFactor(
                imagepaths, context['scale']['size'], util.printPercentage("Extracting Scale Factor"))

            util.printDone("Extracting Scale Factor", True)
            print("Extracted Factor is", context['scale']['factor'])

        if ehanceDataset == '':
            # Step 4: Shuffle dataset
            if 'shuffle' in context and context['shuffle']:
                shuffle(files)
            util.printDone("Shuffling all data")

            # Step 5: Split datasets
            train, dev, test = split.split(
                files, context['dev_frac'], context['test_frac'])
            util.printDone("Splitting data")
        else:
            enhancePath = os.path.join(util.OUTPUT_PATH, ehanceDataset)
            train, dev, test = enhance.enhance(files, enhancePath)
            util.printDone('Retrieved data split from other Dataset', True)

        # Step 6: Apply image pipeline to training
        train, train_sizes = pipeline.applyFullPipeline(
            train, context, util.printPercentage("Processing Training Images"), True)
        util.printDone("Processing Training Images", True)

        # Step 7: Apply image pipeline to dev
        dev, dev_sizes = pipeline.applyFullPipeline(
            dev, context, util.printPercentage("Processing Validation Images"), False)
        util.printDone("Processing Validation Images", True)

        # Step 8: Apply image pipeline to train
        test, test_sizes = pipeline.applyFullPipeline(
            test, context, util.printPercentage("Processing Test Images"), False)
        util.printDone("Processing Test Images", True)

        # Step 9: Shuffle training images
        if 'shuffle' in context and context['shuffle']:
            shuffle(train)
        util.printDone("Shuffling Training data")

        max_size = np.max(
            [train_sizes, dev_sizes, test_sizes], axis=0)
        # Step 10: Create printed dataset
        if 'printed' in context:
            os.makedirs(printedpath)
            print_data, print_max_size = generate_print.generate_printed_sampels(
                train, context['printed'], 'invert' in context and bool(context['invert']), printedpath, max_size[1], max_size[0])
            shuffle(print_data)
            max_size = np.max([print_max_size, max_size], axis=0)
            print_train, print_dev, print_test = split.split(
                print_data, context['dev_frac'], context['test_frac'])
            util.dumpJson(basepath, "print_train", print_train)
            util.dumpJson(basepath, "print_dev", print_dev)
            util.dumpJson(basepath, "print_test", print_test)
            util.printDone("Creating Printed Samples")

        # Step 6: Write datasets
        util.dumpJson(basepath, "train", train)
        util.dumpJson(basepath, "dev", dev)
        util.dumpJson(basepath, "test", test)
        util.printDone("Saving split data")

        print("Maximum Extracted size:", max_size)

        # Step 7: Write meta file
        util.dumpJson(basepath, "meta", index.makeIndex(
            context, max_size, 'printed' in context))
        util.printDone("Writing meta file")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Configuration name", default="iam-lines")
    parser.add_argument(
        "--enhance", help="Dataset to enhance", default="")
    args = parser.parse_args()
    context = loadContext(args.config)
    prepareDataset(args.config, context, args.enhance)
