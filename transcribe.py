import sys
import argparse
from executor import Executor
import os
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
        '--subset', help='Subset to transcribe', default='test')
    args = parser.parse_args()

    exc = Executor(args.config, args.dataset, args.legacy_transpose)
    exc.configure(args.gpu, not args.hardplacement, args.logplacement)
    transcriptions = exc.transcribe(args.subset, args.model_date if args.model_date !=
                                    "" else None, args.model_epoch, {
                                        'trans_batch': transcription_hook
                                    })
    print '\n'
    for i in range(len(transcriptions['files'])):
        print '{:100} -> {}'.format(exc.dataset.decompile(transcriptions['trans'][i]), os.path.basename(transcriptions['files'][i]))
