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
from lib.nn.tfunet import TFUnet

MODEL_DATE = "2018-10-10-17-58-01"
MODEL_EPOCH = 19

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument('--hardplacement', help='Disallow Softplacement, default is False',
                        action='store_true', default=False)
    parser.add_argument('--logplacement', help='Log Device placement',
                        action='store_true', default=False)
    parser.add_argument(
        '--paper-note-path', default='../paper-notes/data/words')
    parser.add_argument(
        '--model-date', help='date to continue for', default=MODEL_DATE)
    parser.add_argument('--model-epoch', help='epoch to continue for',
                        default=MODEL_EPOCH, type=int)
    args = parser.parse_args()

    # TRAINING
    LOG_NAME = '{}-{}'.format("separation", args.model_date)
    model_folder = os.path.join(Constants.MODELS_PATH, LOG_NAME)
    models_path = os.path.join(
        model_folder, 'model-{}'.format(args.model_epoch))
    logger = Logger()
    config = Configuration.load(model_folder, "algorithm")
    algorithm = TFUnet(config['algo_config'])

    algorithm.configure(
        slice_width=config['data_config.slice_width'], slice_height=config['data_config.slice_height'])
    executor = Executor(algorithm, True, config, logger=logger)
    dataset = PaperNoteSlices(
        paper_note_path=args.paper_note_path,
        slice_width=config['data_config.slice_width'],
        slice_height=config['data_config.slice_height'],
        binarize=config.default('binary', False),
        filter=False)

    executor.configure(softplacement=not args.hardplacement,
                       logplacement=args.logplacement, device=args.gpu)

    executor.restore(models_path)
    executables = [SeparationValidator(logger=logger, config=config,
                                       dataset=dataset, subset="dev", prefix="dev", exit_afterwards=True, max_batch_active=False),
                   SeparationValidator(logger=logger, config=config,
                                       dataset=dataset, subset="test", prefix="test", exit_afterwards=True, max_batch_active=False)]

    executor(executables)
