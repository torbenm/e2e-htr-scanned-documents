from collections import Counter
from lib.util.file import readJson
import json


BROWN_CORPUS_FILE = "./data/raw/corpus/brown.txt"
LOB_CORPUS_FILE = "./data/raw/corpus/lob.txt"

OUTPUT = "./data/output/corpus_new.json"

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


def remove_test_words(testconfigfile, words_file):
    global DICTIONARY
    testconfig = readJson(testconfigfile)
    with open(words_file, 'r') as lines:
        while True:
            line = lines.readline().strip()
            if not line:
                break
            if line[0] != "#":
                lsplit = line.split(" ")
                if lsplit[1] != "err":
                    code = lsplit[0][:-3]
                    text = ' '.join(lsplit[8:]).strip().lower()
                    if text in DICTIONARY and code in testconfig:
                        if DICTIONARY[text] == 1:
                            del DICTIONARY[text]
                        else:
                            DICTIONARY[text] = DICTIONARY[text] - 1


parse_file(LOB_CORPUS_FILE, " ", "_", "$")
parse_file(BROWN_CORPUS_FILE, " ", "/", "$")
remove_test_words("../paper-notes/data/final/test.json",
                  "./data/raw/iam/ascii/words.txt")

DICTIONARY_OBJ = {
    "punctuation": list(PUNCTUATION),
    "dictionary": DICTIONARY
}

with open(OUTPUT, 'w+') as f:
    json.dump(DICTIONARY_OBJ, f)
