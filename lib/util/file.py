import json


def writeJson(file, data):
    with open(file, 'w+') as f:
        json.dump(data, f)


def readJson(path):
    with open(path, 'r') as f:
        return json.load(f)
