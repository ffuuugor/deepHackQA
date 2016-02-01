# coding: utf-8
"""
created by artemkorkhov at 2016/01/20
"""

import os
import re
import warnings

import nltk
import pandas as pd

from src.utils.book_processing import get_h1_text, get_h_all_text
from src import BASE_DIR





sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


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


def tokenize_sentences(document):
    """ Splits given doc by sentences
    :param document:
    :return:
    """
    return sentence_tokenizer.tokenize(document)


def get_docs(wiki=True, ck12=True, to_unicode=True):
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


def get_questions():
    """ Gets questions and answers data
    :return:
    """
    texts = []
    TS = pd.read_csv(os.path.join(BASE_DIR, "data/training_set.tsv"), sep='\t')
    VS = pd.read_csv(os.path.join(BASE_DIR, "data/validation_set.tsv"), sep='\t')
    for _, row in TS.iterrows():
        texts.extend([row['question'], row['answerA'], row['answerB'], row['answerC'], row['answerD']])
    for _, row in VS.iterrows():
        texts.extend([row['question'], row['answerA'], row['answerB'], row['answerC'], row['answerD']])
    return [unicode(d.decode('utf8')) for d in texts]


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

    return map(process, docs)

