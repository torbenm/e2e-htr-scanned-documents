import argparse
import os
import time

from data.PaperNoteSlices import PaperNoteSlices
from lib.Executor import Executor
from lib.Configuration import Configuration
from lib import Constants
from lib.Logger import Logger
from lib.executables import DnCNNTrainer, Saver
from nn.dncnn import DnCNN
from nn.unet import Unet

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument('--hardplacement', help='Disallow Softplacement, default is False',
                        action='store_true', default=False)
    parser.add_argument('--logplacement', help='Log Device placement',
                        action='store_true', default=False)
    parser.add_argument(
        '--model-date', help='date to continue for', default='')
    parser.add_argument('--model-epoch', help='epoch to continue for',
                        default=0, type=int)
    args = parser.parse_args()

    # TRAINING
    logger = Logger()
    config = Configuration({
        "name": "dncnn",
        "save": 5,
        "max_batches": 100,
        "batch": 7,
        "learning_rate": 0.001})
    # algorithm = DnCNN({
    #     "conv_n": {
    #         "activation": "tanh"
    #     }
    # })
    algorithm = Unet({})
    algorithm.configure(learning_rate=config['learning_rate'])
    executor = Executor(algorithm, True, config, logger=logger)
    dataset = PaperNoteSlices()

    log_name = '{}-{}'.format(config["name"],
                              time.strftime("%Y-%m-%d-%H-%M-%S"))
    models_path = os.path.join(Constants.MODELS_PATH, log_name)

    executor.configure(softplacement=not args.hardplacement,
                       logplacement=args.logplacement, device=args.gpu)
    executables = [
        DnCNNTrainer(logger=logger, config=config, dataset=dataset),
        Saver(logger=Logger,
              foldername=models_path,
              config=config,
              dataset=dataset,
              every_epoch=config['save'])
    ]

    executor(executables)
