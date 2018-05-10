import tensorflow as tf
import util

if __name__ == "main":
    # TODO: Probably better to use some sort of library for argument handling
    with tf.device(util.evaluate_device(sys.argv[1])):
        pass
