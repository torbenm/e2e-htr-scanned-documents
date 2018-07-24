import tensorflow as tf
import sys
import argparse
from executor import Executor, evaluate_device, MODELS_PATH
import os
from nn.dncnn import DnCNN
from config.config import Configuration
import time
import cv2


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
    msg = 'epoch {0} | loss {1:.3f} | accuracy {3:.3f} | time {2:.3f}'.format(str(epoch).zfill(3),
                                                                              loss,
                                                                              time, val_stats['accuracy'])
    sys.stdout.write('\r{:130}\n'.format(msg))
    sys.stdout.flush()


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

    algorithm = DnCNN({})

    config = Configuration({
        "name": "dncnn",
        "save": 5,
        "epochs": 1000,
        "datapath": "data/output/blended"
    })

    log_name = '{}-{}'.format(config["name"],
                              time.strftime("%Y-%m-%d-%H-%M-%S"))
    models_path = os.path.join(MODELS_PATH, log_name)

    if args.gpu != -1:
        print('Setting cuda visible devices to', args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print("Configuring. Softplacement: ", not args.hardplacement,
          "Logplacement:", args.logplacement, "Allow growth:", True)
    sessionConfig = tf.ConfigProto(
        allow_soft_placement=not args.hardplacement, log_device_placement=args.logplacement)
    sessionConfig.gpu_options.allow_growth = True

    with tf.device(evaluate_device(args.gpu)):
        graph = algorithm.build_graph(channels=3)
        with tf.Session(config=sessionConfig) as sess:
            # if date is None:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # else:
            #     self._restore(sess, date, epoch)
            os.makedirs(models_path, exist_ok=True)
            if config.default('save', False) != False:
                saver = tf.train.Saver(max_to_keep=None)
            truthpath = os.path.join(config["datapath"], "truth")
            imgspath = os.path.join(config["datapath"], "img")
            files = os.listdir(imgspath)
            for n in range(config['epochs']):
                for filename in files:
                    print(filename)

                    X = cv2.imread(os.path.join(imgspath, filename))
                    Y = cv2.imread(os.path.join(truthpath, filename))
                    train_dict = {
                        graph['x']: [X],
                        graph['y']: [Y],
                        graph['is_train']: True
                    }
                    print(
                        sess.run([graph['loss'], graph['train_step']], train_dict))
