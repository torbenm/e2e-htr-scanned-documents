from lib.Configuration import Configuration
from data.datasets import identifyDataset
from lib.util.file import readJson
import pylev

DEFAULT_CONFIG = {
    "separator": "|",
    "source": "./data/corpus.json",
}


class UnigramLanguageModel(object):

    def __init__(self, config={}):
        self.config = Configuration(config, DEFAULT_CONFIG)
        self._build_dictionary()

    def _build_dictionary(self):
        corpus = readJson(self.config["source"])
        self.dictionary = corpus["dictionary"]
        self.punctuation = corpus["punctuation"]

    def __call__(self, regions, file):
        return [self._process_region(region) for region in regions if region.text is not None and region.text != '']

    def _process_region(self, region):
        region.text = self.config["separator"].join([self._process_word(
            word) for word in region.text.split(self.config["separator"])])
        return region

    def _process_word(self, word: str):
        if word.lower() in self.punctuation or word.lower() in self.dictionary:
            return word
        word_lower = word.lower()
        new_word = self._get_best_match(word_lower)
        return new_word

    def _get_case_map(self, word):
        word_lower = word.lower()
        return [1 if word_lower[idx] != word[idx] else 0 for idx in range(len(word))], word_lower

    def _apply_case_map(self, word, word_map):
        return ''.join([word[idx].upper() if word_map[idx] else word[idx] for idx in range(len(word))])

    def _get_best_match(self, word):
        minimum_lev = len(word)
        matches = []
        for new_word in self.dictionary.keys():
            lev = pylev.levenshtein(word, new_word)
            if lev < minimum_lev:
                matches = [new_word]
                minimum_lev = lev
            elif lev == minimum_lev:
                matches.append(new_word)

        max_prob = 0
        final_match = ""
        for match in matches:
            if max_prob < self.dictionary[match]:
                max_prob = self.dictionary[match]
                final_match = match
        return final_match

    def close(self):
        pass


if __name__ == "__main__":

    class test(object):
        text = "An|apple|does|not|fahl|faar|flom|the|trea|!"

    model = UnigramLanguageModel()
    print(model._process_region(test()))
