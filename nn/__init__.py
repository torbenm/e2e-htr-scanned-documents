from puigcerver2017 import Puigcerver2017
from Voigtlaender2016 import VoigtlaenderDoetschNey2016
from graves2009 import GravesSchmidhuber2009
from reshtr import ResHtr


def getAlgorithm(name):
    if name == "puigcerver":
        return Puigcerver2017()
    elif name == "voigtlaender":
        return VoigtlaenderDoetschNey2016()
    elif name == "graves":
        return GravesSchmidhuber2009()
    elif name == "reshtr":
        return ResHtr()
