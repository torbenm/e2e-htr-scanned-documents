import argparse
import os
import time
import cv2
import numpy as np

from data.PaperNoteSlices import PaperNoteSlices
from lib.Executor import Executor
from lib.Configuration import Configuration
from lib import Constants
from lib.Logger import Logger
from lib.executables import SeparationRunner, Saver, SeparationValidator
from nn.unet import Unet

MODEL_DATE = "2018-09-05-11-07-15"
MODEL_EPOCH = 11


def visualize(outputs, X):
    pass
    # for i in range(len(X)):
    #     img = np.argmax(outputs[i], 2)*255.0
    #     img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    #     print(np.min(img))
    #     # print(img)
    #     cv2.imshow('x', X[i])
    #     cv2.imshow('y', img)
    #     cv2.waitKey(0)


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
        "name": "separation",
        "save": 5,
        "max_batches": 1,
        "batch": 5,
        "slice_width": 1024,
        "slice_height": 1024,

    })
    algorithm = Unet({
        "depth": 3,
        "downconv": {
            "filters": 2
        },
        "upconv": {
            "filters": 2
        }
    })
    algorithm.configure(
        slice_width=config['slice_width'], slice_height=config['slice_height'])
    executor = Executor(algorithm, True, config, logger=logger)
    dataset = PaperNoteSlices(
        slice_width=config['slice_width'], slice_height=config['slice_height'])

    log_name = '{}-{}'.format(config["name"], args.model_date)
    models_path = os.path.join(
        Constants.MODELS_PATH, log_name, 'model-{}'.format(args.model_epoch))

    executor.configure(softplacement=not args.hardplacement,
                       logplacement=args.logplacement, device=args.gpu)

    executor.restore(models_path)

    executables = [
        SeparationValidator(logger=logger, config=config,
                            dataset=dataset, after_iter=visualize, subset="dev")
    ]

    executor(executables)
