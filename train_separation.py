import argparse
import os
import time

from data.PaperNoteSlices import PaperNoteSlices
from lib.Executor import Executor
from lib.Configuration import Configuration
from lib import Constants
from lib.Logger import Logger
from lib.executables import SeparationTrainer, Saver, SeparationValidator
from lib.nn.tfunet import TFUnet
from lib.Constants import CONFIG_PATH

SEP_CONFIG_PATH = os.path.join(CONFIG_PATH, "separation")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
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
    config = Configuration.load(SEP_CONFIG_PATH, args.config)
    config()
    algorithm = TFUnet(config['algo_config'])
    algorithm.configure(learning_rate=config['learning_rate'],
                        slice_width=config['data_config.slice_width'], slice_height=config['data_config.slice_height'])
    executor = Executor(algorithm, True, config, logger=logger)
    dataset = PaperNoteSlices(paper_note_path=config.default('data_config.paper_note_path', '../paper-notes/data/final'),
                              filter=config['data_config.filter'],
                              slice_width=config['data_config.slice_width'],
                              slice_height=config['data_config.slice_height'],
                              binarize=config.default('binary', False),
                              config=config['data_config'])

    log_name = '{}-{}'.format(config["name"],
                              time.strftime("%Y-%m-%d-%H-%M-%S"))
    models_path = os.path.join(Constants.MODELS_PATH, log_name)

    executor.configure(softplacement=not args.hardplacement,
                       logplacement=args.logplacement, device=args.gpu)
    executables = [
        SeparationTrainer(logger=logger, config=config, dataset=dataset),
        SeparationValidator(logger=logger, config=config, dataset=dataset),
        Saver(logger=Logger,
              foldername=models_path,
              config=config,
              dataset=dataset,
              every_epoch=config['save'])
    ]
    if args.model_date != "":
        restore_log = '{}-{}'.format(config["name"], args.model_date)
        restore_path = os.path.join(
            Constants.MODELS_PATH, restore_log, 'model-{}'.format(args.model_epoch))
        print("Restoring {}".format(restore_path))
        executor.restore(restore_path)

    executor(executables)
