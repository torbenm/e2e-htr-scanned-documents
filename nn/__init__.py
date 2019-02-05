from nn.puigcerver2017 import Puigcerver2017
from nn.htrnet import HtrNet


def getAlgorithm(name, algoConfig, transpose=False):
    if name == "puigcerver":
        return Puigcerver2017(algoConfig, transpose)
    elif name == "htrnet":
        return HtrNet(algoConfig)
