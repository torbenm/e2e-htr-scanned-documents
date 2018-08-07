from .. import util
import os


def enhance(data, enhancePath):
    devConfig = parseSplit(util.loadJson(enhancePath, "dev"))
    testConfig = parseSplit(util.loadJson(enhancePath, "test"))
    trainConfig = parseSplit(util.loadJson(enhancePath, "train"))
    train = []
    dev = []
    test = []
    for part in data:
        bname = os.path.basename(part['path'])
        if bname in trainConfig:
            train.append(part)
        elif bname in testConfig:
            test.append(part)
        elif bname in devConfig:
            dev.append(part)
    return train, dev, test


def paperNotes(data, paperNotesFolder):
    dev_config = util.loadJson(paperNotesFolder, "dev")
    train_config = util.loadJson(paperNotesFolder, "train")
    test_config = util.loadJson(paperNotesFolder, "test")
    train = []
    dev = []
    test = []
    for file in data:
        if file["line"] in train_config:
            train.append(file)
        elif file["line"] in dev_config:
            dev.append(file)
        elif file["line"] in test_config:
            test.append(file)
    return train, dev, test


def parseSplit(items):
    def _getRawName(item):
        basename = os.path.basename(item['path'])
        splex = os.path.splitext(basename)
        real_name = '_'.join(splex[0].split('_')[:-1])
        return real_name + splex[1]
    return [_getRawName(item) for item in items]
