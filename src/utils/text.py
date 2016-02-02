# coding: utf-8
"""
created by artemkorkhov at 2016/01/20
"""

import os
import re
import warnings
import json

import nltk
import pandas as pd
from itertools import chain

from gensim.utils import smart_open
from gensim.corpora.wikicorpus import filter_wiki, _extract_pages

from src.utils.book_processing import get_h1_text, get_h_all_text
from src import BASE_DIR


def qa_pair(question, answer_option):
    """ Joins question text with answer option.
    :param question:
    :param answer_option:
    :return:
    """
    space_left = True
    space_right = True
    if '_' in question:
        l, r = question.find("_"), question.rfind("_")
        if l > 0 and question[l-1] != " ":
            space_left = False
        if l < len(question) - 2 and question[r+1] != " ":
            space_right = False

        substr = "{sl}{txt}{sr}".format(
            sl="" if space_left else " ",
            txt=answer_option,
            sr="" if space_right else " "
        )
        q = question[:l] + substr + question[r+1:]
    else:
        q = question + " " + answer_option
    return q


def get_docs(wiki=True, ck12=True, raw=True):
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
                    result.append(doc)
        return result

    def parse_book(dirname="data/ck12_book/OEBPS"):
        result = []
        for filename in os.listdir(os.path.join(BASE_DIR, dirname)):
            if filename.endswith('.html'):
                _fn = filename.split('.')[0]
                if _fn.isdigit():
                    with open(os.path.join(BASE_DIR, dirname, filename)) as f:
                        doc = f.read()
                        # result.append({
                        #     "headers": get_h1_text(doc),
                        #     "content": get_h_all_text(doc)
                        # })
                        headers = get_h1_text(doc)
                        contents = get_h_all_text(doc)
                        headers_data = [' '.join([k, v]) for k, v in headers.iteritems()]
                        contents_data = [' '.join([k, ' '.join(v)]) for k, v in contents.iteritems()]
                        result.extend(headers_data)
                        result.extend(contents_data)
        return result

    wiki = parse_dir("data/parsed_wiki_data")
    book = parse_book()
    return wiki, book


def wiki_docs(dir="data/simple_wiki"):
    """
    :param path:
    :return:
    """
    for filename in os.listdir(os.path.join(BASE_DIR, dir)):
        with open(os.path.join(BASE_DIR, dir, filename)) as f:
            doc = filter_wiki(f.read())
            yield doc


def get_swiki(path="data/swiki.json"):
    """ Uses presaved simple wiki
    :param path:
    :return:
    """
    with open(os.path.join(BASE_DIR, path)) as f:
        return json.loads(f.read())


def get_questions():
    """ Gets questions and answers data
    :return:
    """
    texts = []
    TS = pd.read_csv(os.path.join(BASE_DIR, "data/training_set.tsv"), sep='\t')
    VS = pd.read_csv(os.path.join(BASE_DIR, "data/validation_set.tsv"), sep='\t')
    for _, row in TS.iterrows():
        texts.append([row['question'], row['answerA'], row['answerB'], row['answerC'], row['answerD']])
    for _, row in VS.iterrows():
        texts.append([row['question'], row['answerA'], row['answerB'], row['answerC'], row['answerD']])
    # return [unicode(di.decode('utf8')) for d in texts for di in d]
    result = []
    for text in texts:
        result.append([unicode(d.decode('utf8')) for d in text])
    return result


def make_texts(docs, include_questions=True, single=True):
    """ Creates text doc for LDA2vec from documents
    :param docs:
    :param include_questions:
    :param single:
    :return:
    """
    _texts = []
    for doc in docs['wiki']:
        # text = u'{} {}'.format(doc['name'], doc['content'])
        _texts.append(doc['content'])
    for doc in docs['ck12']:
        headers = ' '.join([' '.join(i) for i in doc['headers'].items()])
        contents = ' '.join([' '.join(i) for i in doc['content'].items()])
        text = u'{} {}'.format(headers, contents)
        _texts.append(text)
    if include_questions:
        questions = get_questions()
        _texts.extend(questions)
    return ' '.join(_texts) if single else _texts


def preprocess_text(text, regexp="[^a-zA-Z0-9,\?\.]"):
    """ Performs initial text preprocessing
    :param text:
    :return:
    """
    if isinstance(text, basestring):
        text = [text]
    elif isinstance(text, list):
        pass
    else:
        warnings.warn("text is not a string and not a list of strings! Given: %s" % type(text))

    docs = []

    for doc in text:
        paragraphs = [par for par in doc.split("\n\n\n") if par]
        docs.extend(paragraphs)

    process = lambda x: ' '.join(re.sub(regexp, " ", x).lower().split())

    return map(unicode, map(process, docs))

