from .iam import IamDataset
from .real import RealDataset
from .rimes import RimesDataset


def identifyDataset(name):
    if name == "iam":
        return IamDataset()
    elif name == "real":
        return RealDataset()
    elif name == "rimes":
        return RimesDataset()
