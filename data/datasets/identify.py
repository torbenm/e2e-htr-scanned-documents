from iam import IamDataset


def identifyDataset(name):
    if name == "iam":
        return IamDataset()
