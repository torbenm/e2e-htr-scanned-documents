from load import load


def get_step(name):
    steps = {
        "load": load,
    }
    return steps[name]
