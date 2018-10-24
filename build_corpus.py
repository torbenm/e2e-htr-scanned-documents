from collections import Counter
import json


BROWN_CORPUS_FILE = "./data/raw/corpus/brown.txt"
LOB_CORPUS_FILE = "./data/raw/corpus/lob.txt"

OUTPUT = "./data/output/corpus.json"

DEV_TEXT_FILE = ""
TEST_TEXT_FILE = ""

DICTIONARY = {}
PUNCTUATION = set()


def parse_file(filepath, word_separator, tag_separator, end_tag):
    global PUNCTUATION, DICTIONARY
    with open(filepath, 'r') as f:
        for line in f.readlines():
            for wordtag in line.split(word_separator):
                wordsplit = wordtag.lower().split(tag_separator)
                if len(wordsplit) != 2:
                    continue
                word, tag = wordsplit
                if tag[-1] == "$":
                    tag = tag[:-1]
                if word == tag and not word.isalpha():
                    PUNCTUATION.add(word)
                else:
                    DICTIONARY[word] = 1 if word not in DICTIONARY else DICTIONARY[word] + 1


parse_file(LOB_CORPUS_FILE, " ", "_", "$")
parse_file(BROWN_CORPUS_FILE, " ", "/", "$")

dictionary_obj = {
    "punctuation": list(PUNCTUATION),
    "dictionary": DICTIONARY
}

with open(OUTPUT, 'w+') as f:
    json.dump(dictionary_obj, f)
