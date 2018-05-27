import abc
import util
import os


class Dataset(object):

    def __init__(name):
        self.name = name
        self.datapath = os.path.join(util.OUTPUT_PATH, name)
        self._load_vocab()
        self._load_sets()
        self._calc_max_length()
        self._compile_sets()

    def _load_meta(self):
        self.meta = util.loadJson(self.datapath, "meta")

    def _load_vocab(self):
        self.vocab = util.loadJson(self.datapath, "vocab")
        self.vocab_length = len(self._vocab[0])

    def _load_sets(self):
        self.data = {
            "train": util.loadJson(self.datapath, "train"),
            "dev": util.loadJson(self.datapath, "dev"),
            "test": util.loadJson(self.datapath, "test")
        }

    def _compile_set(self, dataset):
        for item in self.data[dataset]:
            item['compiled'] = self.compile(item['truth'])

    def _compile_sets(self):
        self._compile_set("train")
        self._compile_set("dev")
        self._compile_set("test")

    def _calc_max_length(self):
        _all = []
        _all.extend(self.data["train"])
        _all.extend(self.data["test"])
        _all.extend(self.data["dev"])
        self.max_length = max(map(lambda x: len(x["truth"]), _all))

    def compile(self, text):
        parsed = [self.vocab[1][c] for c in text]
        parsed.extend([-1] * (self.max_length - len(text)))
        return parsed

    def decompile(self, values):
        def getKey(key):
            try:
                return self.vocab[0][str(c)]
            except KeyError:
                return ''
        return ''.join([getKey(c) for c in values])

    def generateBatch(self, batch_size, max_batches=0, dataset="train"):
        num_batches = self.getBatchCount(batch_size, max_batches, set)
        for b in range(num_batches - 1):
            x = self._raw_x[b * batch_size:(b + 1) * batch_size]
            y = self._raw_y[b * batch_size:(b + 1) * batch_size]
            l = self._raw_l[b * batch_size:(b + 1) * batch_size]
            yield x, y, l
        pass

    def generateEpochs(self, batch_size, num_epochs, max_batches=0, dataset="train"):
        for e in range(num_epochs):
            yield self.generateBatch(batch_size, max_batches=max_batches, dataset=dataset)

    def getBatchCount(self, batch_size, max_batches=0, dataset="train"):
        total_len = len(self.data[dataset])
        num_batches = total_len // batch_size
        return min(
            num_batches, max_batches) if max_batches > 0 else num_batches

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generateEpochs(self, batch_size, num_epochs):
        pass

    @abc.abstractmethod
    def prepareDataset(self, validation_size=0, test_size=0, shuffle=False):
        pass

    @abc.abstractmethod
    def getValidationSet(self):
        pass

    @abc.abstractmethod
    def compile(self, text):
        pass

    @abc.abstractmethod
    def decompile(self, array):
        pass

    @abc.abstractmethod
    def maxLength(self):
        pass

    @abc.abstractmethod
    def getBatchCount(self):
        pass
