import argparse
import os
import time
import cv2
import numpy as np

from data.PaperNoteSlices import PaperNoteSlices
from lib.Executor import Executor
from lib.Configuration import Configuration
from lib.Constants import Constants
from lib.Logger import Logger
from lib.executables import DnCNNRunner, Saver
from nn.dncnn import DnCNN

MODEL_DATE = "2018-08-14-17-18-59"
MODEL_EPOCH = 19


def visualize(outputs, X):
    for i in range(len(X)):
        print(i)
        print(np.mean(outputs[i]))
        cv2.imshow('x', X[i])
        cv2.imshow('y', outputs[i])
        cv2.waitKey(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument('--hardplacement', help='Disallow Softplacement, default is False',
                        action='store_true', default=False)
    parser.add_argument('--logplacement', help='Log Device placement',
                        action='store_true', default=False)
    parser.add_argument(
        '--model-date', help='date to continue for', default=MODEL_DATE)
    parser.add_argument('--model-epoch', help='epoch to continue for',
                        default=MODEL_EPOCH, type=int)
    args = parser.parse_args()

    # TRAINING
    logger = Logger()
    config = Configuration({
        "name": "dncnn",
        "save": 5,
        "max_batches": 10,
        "batch": 5
    })
    algorithm = DnCNN({})
    algorithm.configure()
    executor = Executor(algorithm, True, config, logger=logger)
    dataset = PaperNoteSlices()

    log_name = '{}-{}'.format(config["name"], args.model_date)
    models_path = os.path.join(
        Constants.MODELS_PATH, log_name, 'model-{}'.format(args.model_epoch))

    executor.configure(softplacement=not args.hardplacement,
                       logplacement=args.logplacement, device=args.gpu)

    executor.restore(models_path)

    executables = [
        DnCNNRunner(logger=logger, config=config,
                    dataset=dataset, after_iter=visualize, subset="train")
    ]

    executor(executables)
