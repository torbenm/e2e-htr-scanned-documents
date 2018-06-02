import sys
import argparse
from executor import Executor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def val_batch_hook(step, max_steps, val_stats):
    percent = int((float(step) / max_steps) * 100)
    msg = 'VALIDATING... {:2} %'.format(percent)
    sys.stdout.write('\r{:130}'.format(msg))
    sys.stdout.flush()


def compare_outputs(executor, pred, actual):
    pred = executor.dataset.decompile(pred)
    actual = executor.dataset.decompile(actual)
    out = '{:' + str(executor.dataset.max_length) + '}  {}'
    return out.format(pred, actual)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model-date', default="")
    parser.add_argument('--model-epoch', default=0)
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument('--hardplacement', help='Allow Softplacement, default is True',
                        action='store_true', default=True)
    parser.add_argument('--logplacement', help='Log Device placement',
                        action='store_true', default=False)
    parser.add_argument(
        '--examples', help='Number of examples to display', default=10)
    args = parser.parse_args()

    exc = Executor(args.config)
    exc.configure(args.gpu, not args.hardplacement, args.logplacement)
    results = exc.validate(args.model_date if args.model_date !=
                           "" else None, args.model_epoch, {
                               'val_batch': val_batch_hook
                           })
    print "LER for Validation dataset was:", results['ler']
    len_results = len(results['examples']['trans'])
    ex_length = len_results if args.examples == - \
        1 else min(args.examples, len_results)
    print "Going to show", ex_length, "example transcriptions"
    print '\n'.join([compare_outputs(exc, results['examples']['trans'][c], results['examples']['Y'][c]) for c in range(ex_length - 1)])
