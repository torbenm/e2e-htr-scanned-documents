import tensorflow as tf
from typing import Union, List
import os
from functools import reduce

from lib.nn.layer.algorithmBase import AlgorithmBase
from nn import getAlgorithm

from .executables import Executable
from .Configuration import Configuration

INITIALIZED_SESSION = "session"
INITIALIZED_DEVICE = "device"


class Executor(object):

    DEFAULT_CONFIG = {
        "verbose": False,
        "device": -1,
        "softplacement": True,
        "logplacement": False,
        "allowGrowth": True
    }

    initialized = []
    session = None
    graph = None
    restore_model = None
    g = None

    def __init__(self, algorithm: AlgorithmBase, verbose=False, config={}, logger=None):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.algorithm = algorithm
        self.config = Configuration(config, self.DEFAULT_CONFIG)
        self.verbose = verbose
        self.logger = logger
    #
    # PUBLIC METHODS
    #

    def log(self, message: str, force=False):
        if self.verbose or force:
            print(message)

    def restore(self, model):
        self.restore_model = model

    def configure(self, **config):
        self.config.set(config)
        self.algorithm.set_cpu(self.config['device'] == -1)

    def __call__(self, executables, new_session=False, auto_close=True):
        with tf.device(self._get_device()):
            if self.g is None:
                self.g = tf.Graph()
            with self.g.as_default():
                self._create_graph()
                [executable.extend_graph(self.graph)
                 for executable in executables]
                self._get_session(new_session)
                self._initialize_session()
                self._run(executables)
                if auto_close:
                    self.close()

    def close(self):
        if self.session is not None:
            self.session.close()

    #
    # PRIVATE METHODS
    #

    def _run(self, executables):
        epoch = 0
        while reduce(lambda a, e: a and e, [e.will_continue(epoch) for e in executables]):
            running_executables = filter(
                lambda e: e.will_run(epoch), executables)
            [e(self, epoch, self.session, self.graph)
             for e in running_executables]
            self._summary(epoch, executables)
            epoch += 1

    def _create_graph(self):
        if self.graph is None:
            self.graph = self.algorithm.build_graph()
        return self.graph

    def _get_session(self, new_session=False) -> tf.Session:
        if self.session is None or new_session:
            softplacement = self.config["softplacement"]
            logplacement = self.config["logplacement"]
            allow_growth = self.config["allowGrowth"]
            sessionConfig = tf.ConfigProto(
                allow_soft_placement=softplacement, log_device_placement=logplacement)
            sessionConfig.gpu_options.allow_growth = allow_growth
            self.log("Configuring. Softplacement: {}, Logplacement: {}, Allow growth: {}".format(
                softplacement, logplacement, allow_growth))
            self.session = tf.Session(config=sessionConfig)
            if INITIALIZED_SESSION in self.initialized:
                self.initialized.remove(INITIALIZED_SESSION)
        return self.session

    def _initialize_session(self):
        if INITIALIZED_SESSION not in self.initialized:
            if self.restore_model is None:
                self.session.run(tf.global_variables_initializer())
                self.session.run(tf.local_variables_initializer())
            else:
                tf.train.Saver().restore(self.session, self.restore_model)
            self.initialized.append(INITIALIZED_SESSION)

    def _get_device(self) -> str:
        device = self.config["device"]
        if INITIALIZED_DEVICE not in self.initialized and device != -1:
            self.log('Setting cuda visible devices to %s' % device)
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
            self.initialized.append(INITIALIZED_DEVICE)
        return "/device:CPU:0" if device == -1 else "/device:GPU:{}".format(device)

    def _summary(self, epoch, executables):
        if self.logger is not None:
            exec_time = reduce(lambda a, x: a + x,
                               [e.execution_time for e in executables])
            summary = {
                "time": exec_time
            }

            [e.summarize(summary) for e in executables]
            self.logger.summary("Epoch {:4d}".format(epoch), summary)
