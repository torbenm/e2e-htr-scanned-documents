import sys
import argparse
from executor import Executor
import os
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def transcription_hook(step, max_steps):
    percent = int((float(step) / max_steps) * 100)
    msg = 'TRANSCRIBING... {:2} %'.format(percent)
    sys.stdout.write('\r{:130}'.format(msg))
    sys.stdout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model-date', default="")
    parser.add_argument('--model-epoch', default=0)
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument('--hardplacement', help='Allow Softplacement, default is True',
                        action='store_true', default=False)
    parser.add_argument('--logplacement', help='Log Device placement',
                        action='store_true', default=False)
    parser.add_argument('--legacy-transpose', help='Legacy: Perform transposing',
                        action='store_true', default=False)
    parser.add_argument(
        '--dataset', help='Dataset to transcribe', default='real-iam-lines')

    parser.add_argument(
        '--image', help='Path to image to read')
    args = parser.parse_args()

    exc = Executor(args.config, args.dataset, args.legacy_transpose)
    exc.configure(args.gpu, not args.hardplacement, args.logplacement)
    activations = exc.visualize(args.image, args.model_date if args.model_date !=
                                "" else None, args.model_epoch)

    def scale_range(inp, _min, _max):
        inp += -(np.min(inp))
        inp /= np.max(inp) / (_max - _min)
        inp += _min
        return inp

    conv_layer = 0
    for activation in activations:
        features = activation.shape[3]
        for feature in range(features):
            act_map = activation[0, :, :, feature]
            cv2.imshow('layer {} for feature {}'.format(
                conv_layer, feature), act_map)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        conv_layer += 1
