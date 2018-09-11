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
from nn.tfunet import TFUnet

MODEL_DATE = "2018-09-10-23-05-06"
MODEL_EPOCH = 86


def visualize(outputs, X):
    for i in range(len(X)):
        img = np.argmax(outputs[i], 2)
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        # print(img)
        cv2.imshow('x', X[i])
        cv2.imshow('y', img*255.0)
        cv2.imshow('o', (255-(1-img)*(255-X[i])))
        cv2.waitKey(0)
    pass


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
        "max_batches": {
            "sep": {
                "train": 300,
                "val": 10,
                "pred": 0
            }
        },
        "slice_width": 512,
        "slice_height": 512,
        "batch": 8,
        "learning_rate": 0.001
    })
    algorithm = TFUnet({})

    algorithm.configure(
        slice_width=config['slice_width'], slice_height=config['slice_height'])
    executor = Executor(algorithm, True, config, logger=logger)
    dataset = PaperNoteSlices(
        slice_width=config['slice_width'], slice_height=config['slice_height'], filter=False, single_page=True)
    log_name = '{}-{}'.format(config["name"], args.model_date)
    models_path = os.path.join(
        Constants.MODELS_PATH, log_name, 'model-{}'.format(args.model_epoch))

    executor.configure(softplacement=not args.hardplacement,
                       logplacement=args.logplacement, device=args.gpu)

    executor.restore(models_path)
    runner = SeparationRunner(logger=logger, config=config,
                              dataset=dataset, subset="dev")
    executables = [runner]

    for idx in range(10):
        dataset.next_file("dev")
        executor(executables,  auto_close=False)
        original = cv2.imread(dataset.file['paper'], cv2.IMREAD_GRAYSCALE)
        outputs = np.argmax(np.asarray(runner.outputs), 3)
        merged = dataset.merge_slices(outputs, original.shape)

        allofit = np.concatenate(
            (original, merged*255, (255-(1-merged)*(255-original))), axis=1)
        cv2.imwrite('prediction/output_{}.png'.format(idx), allofit)
