import argparse
from data.iam import IamDataset


def get_dataset(args):
    dataset = None
    if args.dataset == 'iam':
        dataset = IamDataset(args.binarize, args.width,
                             args.height, args.label_padding)


parser = argparse.ArgumentParser()

parser.add_argument('--train', help='Activates training mode',
                    action='store_true', default=False)
parser.add_argument('--gpu', help='Runs scripts on gpu. Default is cpu.',
                    action='store_true', default=False)
parser.add_argument('--prepare-dataset', help='Activates dataset preparation mode',
                    action='store_true', default=False)
parser.add_argument('--binarize', help='Whether dataset should be binarized',
                    action='store_true', default=False)
parser.add_argument('--dataset', help='Which dataset to use',
                    default='iam', type=str)
parser.add_argument('--algorithm', help='Which algorithm to use',
                    default='graves2009', type=str)
parser.add_argument('--width', help='Width of image', default=300, type=int)
parser.add_argument('--label-padding',
                    help='Padding of labels', default=5, type=int)
parser.add_argument('--height', help='Height of image', default=30, type=int)
parser.add_argument('--batch', help='Batch size', default=256, type=int)
parser.add_argument('--epochs', help='Number of epochs', default=100, type=int)
parser.add_argument(
    '--segmentation', help='Which segmentation to use', default='lines', type=str)
parser.add_argument(
    '--pwd', help='Password to log into dataset, if necessary', default='user', type=str)
parser.add_argument(
    '--user', help='User name to log into dataset, if necessary', default='pwd', type=str)

args = parser.parse_args()

dataset = get_dataset(args)

if args.train:
