# coding: utf-8
"""
created by artemkorkhov at 2016/02/01
"""

import os
import re
import logging

import pandas as pd
from lda2vec import preprocess, LDA2Vec, Corpus

from src import BASE_DIR
from src.book_processing import get_h1_text, get_h_all_text


TS = pd.read_csv(os.path.join(BASE_DIR, "data/training_set.tsv"), sep='\t')


def get_corpus(wiki=True, ck12=True, to_unicode=True):
    """ Returns all text data that we have for learning models
    :param wiki:
    :param ck12:
    :param to_unicode:
    :return:
    """
    def parse_dir(dirname):
        result = []
        for filename in os.listdir(os.path.join(BASE_DIR, dirname)):
            if not filename.endswith('.json'):
                with open(os.path.join(BASE_DIR, dirname, filename)) as fn:
                    doc = fn.read()
                    if to_unicode:
                        doc = unicode(doc)
                    result.append({
                        "name": filename.split('.')[0],
                        "content": doc
                    })
        return result

    def parse_book(dirname="data/ck12_book/OEBPS"):
        result = []
        for filename in os.listdir(os.path.join(BASE_DIR, dirname)):
            if filename.endswith('.html'):
                _fn = filename.split('.')[0]
                if _fn.isdigit():
                    with open(os.path.join(BASE_DIR, dirname, filename)) as f:
                        doc = f.read()
                        if to_unicode:
                            doc = unicode(doc)
                        result.append({
                            "headers": get_h1_text(doc),
                            "content": get_h_all_text(doc)
                        })
        return result

    texts = {}
    if wiki:
        texts['wiki'] = parse_dir("data/parsed_wiki_data")
    if ck12:
        texts['ck12'] = parse_book()

    return texts


def make_corpus(docs):
    """ Creates corpus for LDA2vec from documents
    :param dict docs:
    :return:
    """



if __name__ == '__main__':
    texts = get_corpus()
    # print len(texts['wiki'])
    # print len(texts['ck12'])
    print texts['ck12'][0]['headers'].keys()
    print texts['ck12'][0]['content'].keys()

