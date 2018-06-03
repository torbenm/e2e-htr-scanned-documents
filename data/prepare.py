import os
import util
from datasets import identifyDataset
from steps import vocab, scalefactor, pipeline, split, index
import sys
import numpy as np
from random import shuffle


def loadContext(name):
    return util.loadJson(util.CONFIG_PATH, name)


def prepareDataset(name, context):
    dataset = identifyDataset(context['dataset'])
    if dataset is not None:
        basepath = os.path.join(util.OUTPUT_PATH, name)
        imagepath = os.path.join(basepath, "imgs")

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
        if 'scale' in context:
            imagepaths = map(lambda x: x["path"], files)
            context['scale']['factor'] = scalefactor.getScaleFactor(
                imagepaths, context['scale']['size'], util.printPercentage("Extracting Scale Factor"))

            util.printDone("Extracting Scale Factor", True)

        # Step 4: Shuffle dataset
        if 'shuffle' in context and context['shuffle']:
            shuffle(files)
        util.printDone("Shuffling all data")

        # Step 5: Split datasets
        train, dev, test = split.split(
            files, context['dev_frac'], context['test_frac'])
        util.printDone("Splitting data")

        # Step 6: Apply image pipeline to training
        train = pipeline.applyFullPipeline(
            train, context, util.printPercentage("Processing Training Images"), True)
        util.printDone("Processing Training Images", True)

        # Step 7: Apply image pipeline to dev
        dev = pipeline.applyFullPipeline(
            dev, context, util.printPercentage("Processing Validation Images"), False)
        util.printDone("Processing Validation Images", True)

        # Step 7: Apply image pipeline to train
        test = pipeline.applyFullPipeline(
            test, context, util.printPercentage("Processing Test Images"), False)
        util.printDone("Processing Test Images", True)

        # Step 4: Shuffle training images
        if 'shuffle' in context and context['shuffle']:
            shuffle(train)
        util.printDone("Shuffling Training data")

        # Step 6: Write datasets
        util.dumpJson(basepath, "train", train)
        util.dumpJson(basepath, "dev", dev)
        util.dumpJson(basepath, "test", test)
        util.printDone("Saving split data")

        # Step 7: Write meta file
        util.dumpJson(basepath, "meta", index.makeIndex(context))
        util.printDone("Writing meta file")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Configuration name", default="iam-lines")
    args = parser.parse_args()
    context = loadContext(args.config)
    prepareDataset(args.config, context)
