# coding: utf-8
"""
created by artemkorkhov at 2016/01/16
"""
import os
import signal
import re
import json
import itertools
import pandas as pd

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords


ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'

POS = {
    'NOUN': 'n',
    'ADJ': 'a',
    'ADJ_SAT': 's',
    'ADV': 'r',
    'VERB': 'v'
}

synset = {}


DIR = os.path.dirname(__file__)
BASE_DIR = "/Users/artemkorkhov/Projects/kaggle/deephack"


def tokenize(review, remove_stopwords = True ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    # 1. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review)
    # 2. Convert words to lower case and split them
    words = review_text.lower().split()
    # 3. Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    # 5. Return a list of words
    return words


def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def make_synlists(tokens, depth=2):
    """
    :param document:
    :return:
    """
    words_synonyms = {}

    for word in tokens:
        synonyms = set()
        if word not in words_synonyms:
            _, pos = nltk.pos_tag([word], tagset='universal')[0]
            wn_pos = POS.get(pos)
            if wn_pos:
                synlist = wn.synsets(word, pos=wn_pos)
            else:
                synlist = wn.synsets(word)
            for synset in synlist[:depth]:
                for w in synset.lemma_names():
                    synonyms.add(w)
            words_synonyms.setdefault(word, list(synonyms))
    print len(list(itertools.chain.from_iterable(words_synonyms.values())))
    return words_synonyms


def save(signum, frame):
    with open("/Users/artemkorkhov/Projects/kaggle/deephack/source_docs/synlist_correct.json", "w+") as f:
        json.dump(synset, f)


def synonymize_questions():
    """ make synonyms for wiki words
    """
    global synset
    # for txt in os.listdir(os.path.join(BASE_DIR, dir_)):
    #     with open(os.path.join(BASE_DIR, dir_, txt)) as f:
    #         doc = f.read()
    #         print len(doc)
    #         tokens = filter(lambda x: x not in synset, tokenize(doc))
    #         print len(tokens)
    #         synset.update(make_synlists(tokens, depth=2))
    # with open("/Users/artemkorkhov/Projects/kaggle/deephack/source_docs/synlist.json", "w+") as f:
    #     json.dump(synset, f)

    questions = pd.read_csv(
        "/Users/artemkorkhov/Projects/kaggle/deephack/data/validation_set.tsv", sep='\t'
    )
    for index, row in questions.iterrows():
        print "doc number ==>", index
        doc = row["question"]+' '+row["answerA"]+' '+row["answerB"]+' '+row["answerC"]+' '+row["answerD"]
        print len(doc)
        tokens = filter(lambda x: x not in synset, tokenize(doc))
        print len(tokens)
        synset.update(make_synlists(tokens, depth=2))
    with open("/Users/artemkorkhov/Projects/kaggle/deephack/source_docs/synlist_correct.json", "w+") as f:
        json.dump(synset, f)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, save)
    signal.signal(signal.SIGTERM, save)
    synonymize_questions()