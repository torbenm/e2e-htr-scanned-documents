from lib.nn.htrnet import HtrNet


def getAlgorithm(name, algoConfig, transpose=False):
    if name == "htrnet":
        return HtrNet(algoConfig)
