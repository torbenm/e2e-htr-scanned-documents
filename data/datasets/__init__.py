from .iam import IamDataset
from .real import RealDataset


def identifyDataset(name):
    if name == "iam":
        return IamDataset()
    elif name == "real":
        return RealDataset()
