import sys
import argparse
import os
from lib.QuickExecutor import QuickExecutor

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
        '--subset', help='Subset to transcribe', default='test')
    args = parser.parse_args()

    qe = QuickExecutor(args.dataset, args.config)
    qe.configure(softplacement=not args.hardplacement,
                 logplacement=args.logplacement, device=args.gpu)
    transcriber = qe.add_transcriber(subset=args.subset)
    qe.restore(args.model_date, args.model_epoch)
    qe()
    transcriptions = transcriber.transcriptions

    print('\n')
    print('Transcriptions for {} lines'.format(len(transcriptions['files'])))
    max_trans_l = max(map(lambda t: len(t), transcriptions['trans']))
    line_format = '{0:'+str(max_trans_l+10)+'} {1:30} {2}'
    heading = line_format.format('Transcription', 'Classification', 'File')
    print(heading)
    print("-"*len(heading))
    for i in range(len(transcriptions['files'])):
        decompiled = qe.dataset.decompile(transcriptions['trans'][i])
        filename = os.path.basename(transcriptions['files'][i])
        if len(transcriptions['class']) > i:
            is_ht = 'Handwritten' if transcriptions['class'][i][0] > 0.5 else 'Printed'
            is_ht = '{:12} ({:05.2f} %)'.format(
                is_ht, transcriptions['class'][i][0]*100)

        else:
            is_ht = '?'

        print(line_format.format(decompiled, is_ht, filename))
