import argparse
import os
import time
import cv2
import numpy as np

from data.PaperNoteSlices import PaperNoteSlices
from data import util, Dataset, PreparedDataset
from lib.Executor import Executor
from lib.Configuration import Configuration
from lib import Constants
from lib.Logger import Logger
from lib.executables import RecognitionValidator, ClassValidator
from data.PaperNoteWords import PaperNoteWords
from lib.nn.htrnet import HtrNet


MODEL_DATE = "2018-10-30-13-33-59"
MODEL_EPOCH = 74

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
    parser.add_argument(
        '--paper-note-path', default='../paper-notes/data/words')
    parser.add_argument('--model-epoch', help='epoch to continue for',
                        default=MODEL_EPOCH, type=int)
    args = parser.parse_args()

    # TRAINING
    LOG_NAME = '{}-{}'.format("otf-iam-paper", args.model_date)
    model_folder = os.path.join(Constants.MODELS_PATH, LOG_NAME)
    models_path = os.path.join(
        model_folder, 'model-{}'.format(args.model_epoch))
    logger = Logger()
    config = Configuration.load(model_folder, "algorithm")
    algorithm = HtrNet(config['algo_config'])
    dataset = PreparedDataset.PreparedDataset(config[
        'dataset'], False, config['data_config'])

    algorithm.configure(batch_size=config['batch'], learning_rate=config[
        'learning_rate'], sequence_length=dataset.max_length,
        image_height=dataset.meta["height"], image_width=dataset.meta[
        "width"], vocab_length=dataset.vocab_length, channels=dataset.channels,
        class_learning_rate=config.default('class_learning_rate', config['learning_rate']))
    executor = Executor(algorithm, True, config, logger=logger)

    paper_note_dataset = PaperNoteWords(
        paper_note_path=args.paper_note_path,
        meta=dataset.meta, vocab=dataset.vocab, data_config=dataset.data_config, max_length=dataset.max_length)

    executor.configure(softplacement=not args.hardplacement,
                       logplacement=args.logplacement, device=args.gpu)

    executor.restore(models_path)
    executables = [RecognitionValidator(logger=logger, config=config,
                                        dataset=dataset, subset="dev", prefix="dev", exit_afterwards=True),
                   RecognitionValidator(logger=logger, config=config,
                                        dataset=dataset, subset="test", prefix="test", exit_afterwards=True),
                   ClassValidator(logger=logger, config=config,
                                  dataset=dataset, subset="print_dev", prefix="dev", exit_afterwards=True),
                   ClassValidator(logger=logger, config=config,
                                  dataset=dataset, subset="print_test", prefix="test", exit_afterwards=True)]

    executables_pn = [RecognitionValidator(logger=logger, config=config,
                                           dataset=paper_note_dataset, subset="dev", prefix="pn dev", exit_afterwards=True),
                      RecognitionValidator(logger=logger, config=config,
                                           dataset=paper_note_dataset, subset="test", prefix="pn test", exit_afterwards=True),
                      ClassValidator(logger=logger, config=config,
                                     dataset=paper_note_dataset, subset="print_dev", prefix="pn dev", exit_afterwards=True),
                      ClassValidator(logger=logger, config=config,
                                     dataset=paper_note_dataset, subset="print_test", prefix="pn test", exit_afterwards=True)]

    executor(executables, auto_close=False)
    executor(executables_pn, auto_close=True)
