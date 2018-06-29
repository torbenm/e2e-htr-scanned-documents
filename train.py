import sys
import argparse
from executor import Executor
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def batch_hook(epoch, batch, max_batches):
    percent = (float(batch) / max_batches) * 100
    out = 'epoch = {0} [ {2:100} ] {1:02.2f}% '.format(
        str(epoch).zfill(3), percent, "|" * int(percent))
    sys.stdout.write("\r" + out)
    sys.stdout.flush()


def val_batch_hook(step, max_steps, val_stats):
    percent = int((float(step) / max_steps) * 100)
    msg = 'VALIDATING... {:2} %'.format(percent)
    sys.stdout.write('\r{:130}'.format(msg))
    sys.stdout.flush()


def epoch_hook(epoch, loss, time, val_stats):
    msg = 'epoch {0} | loss {1:.3f} | val_loss {4:.3f} | cer {3:.3f} | time {2:.3f}'.format(str(epoch).zfill(3),
                                                                                            loss,
                                                                                            time, val_stats['cer'], val_stats['loss'])
    sys.stdout.write('\r{:130}\n'.format(msg))
    sys.stdout.flush()


def class_epoch_hook(epoch, loss, time, val_stats):
    msg = 'epoch {0} | loss {1:.3f} | accuracy {4:.3f} | time {2:.3f}'.format(str(epoch).zfill(3),
                                                                              loss,
                                                                              time, val_stats['accuracy'])
    sys.stdout.write('\r{:130}\n'.format(msg))
    sys.stdout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument(
        '--gpu', help='Runs scripts on gpu. Default is cpu.', default=-1, type=int)
    parser.add_argument('--hardplacement', help='Disallow Softplacement, default is False',
                        action='store_true', default=False)
    parser.add_argument('--logplacement', help='Log Device placement',
                        action='store_true', default=False)
    parser.add_argument('--skip-validation', help='Skip validation',
                        action='store_true', default=False)
    parser.add_argument('--timeline', default='')
    parser.add_argument('--legacy-transpose', help='Legacy: Perform transposing',
                        action='store_true', default=False)
    parser.add_argument(
        '--model-date', help='date to continue for', default='')
    parser.add_argument('--model-epoch', help='epoch to continue for',
                        default=0, type=int)
    args = parser.parse_args()

    exc = Executor(args.config, transpose=args.legacy_transpose)
    exc.configure(args.gpu, not args.hardplacement, args.logplacement)
    exc.train(args.model_date if args.model_date !=
              "" else None, args.model_epoch, {
                  'batch': batch_hook,
                  'val_batch': val_batch_hook,
                  'epoch': epoch_hook
                  'class_epoch': class_epoch_hook
              }, {
                  'skip_validation': args.skip_validation,
                  'timeline': args.timeline
              })
