import sys
import argparse
import os
from lib import QuickExecutor

ALGORITHM_CONFIG = "otf-iam-both"
# "2018-07-07-14-59-06"  # "2018-07-02-23-46-51"
MODEL_DATE = "2018-08-12-23-45-59"
# 800  # 65
MODEL_EPOCH = 24

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=ALGORITHM_CONFIG)
    parser.add_argument('--model-date', default=MODEL_DATE)
    parser.add_argument('--model-epoch', default=MODEL_EPOCH)
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument('--hardplacement', help='Allow Softplacement, default is True',
                        action='store_true', default=False)
    parser.add_argument('--logplacement', help='Log Device placement',
                        action='store_true', default=False)
    parser.add_argument('--legacy-transpose', help='Legacy: Perform transposing',
                        action='store_true', default=False)
    parser.add_argument(
        '--dataset', help='Dataset to transcribe', default='iam-both1-raw')
    parser.add_argument(
        '--subset', help='Subset to transcribe', default='print_test')
    args = parser.parse_args()

    qe = QuickExecutor(args.dataset, args.config)
    qe.config['max_batches'] = 20
    qe.configure(softplacement=not args.hardplacement,
                 logplacement=args.logplacement, device=args.gpu)
    transcriber = qe.add_classifier(subset=args.subset)
    qe.restore(args.model_date, args.model_epoch)
    qe()
    transcriptions = transcriber.transcriptions

    print('\n')
    print('Classification for {} lines'.format(len(transcriptions['files'])))
    line_format = '{0:30} {1:30} {2}'
    heading = line_format.format('Original', 'Classification', 'File')
    print(heading)
    print("-"*len(heading))
    wrong = 0
    for i in range(len(transcriptions['files'])):
        original = 'Handwritten' if transcriptions['original'][i][0] == 1 else 'Printed'
        filename = os.path.basename(transcriptions['files'][i])
        if len(transcriptions['class']) > i:
            is_ht = 'Handwritten' if transcriptions['class'][i][0] > 0.5 else 'Printed'
            is_ht = '{:12} ({:05.2f} %)'.format(
                is_ht, transcriptions['class'][i][0]*100)
        else:
            is_ht = '?'

        if (transcriptions['original'][i][0] == 1) != (transcriptions['class'][i][0] > 0.5):
            wrong += 1

        print(line_format.format(original, is_ht, filename))
    print('Accuracy {:5.3f}'.format(
        1-(wrong/float(len(transcriptions['files'])))))
