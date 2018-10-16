from lib.Configuration import Configuration

DEFAULT_CONFIG = {
    "punctuation": ",.;:!?",
    "split": "|"
}


class BagOfWords(object):

    def __init__(self, config={}, default_config=DEFAULT_CONFIG):
        self.config = Configuration(config, default_config)

    def _append_words(self, collection, words):
        total = 0
        for word in words:
            if word not in self.config["punctuation"]:
                collection[word] = 1 if word not in collection else collection[word] + 1
                total += 1
        return total

    def withinfo(self, groundtruth, predictions):
        total = 0
        correct = 0
        truth_words = {}
        pred_words = {}
        for gt in groundtruth:
            total += self._append_words(
                truth_words, gt.text.lower().split(self.config["split"]))
        for pred in predictions:
            self._append_words(pred_words, pred.text.lower().split(
                self.config["split"]))

        for word in truth_words.keys():
            if word in pred_words:
                correct += min(pred_words[word], truth_words[word])

        if total == 0:
            return [1]

        return [{"bow": correct / total}]

    def __call__(self, groundtruth, predictions):
        return self.withinfo(groundtruth, predictions)[0]
