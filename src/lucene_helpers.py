# coding: utf-8
"""
created by artemkorkhov at 2016/01/26
"""

import os
import sys
import re

import lucene
import pandas as pd
import numpy as np

import book_processing
from src import BASE_DIR
# import utils
from lucene import *
from src.make_synonyms import tokenize

# lucene.initVM()

from org.apache.lucene.document import Document, Field
from org.apache.lucene.analysis.snowball import SnowballAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.util import Version
from org.apache.lucene.index import IndexWriter, IndexWriterConfig

from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import IndexReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.shingle import ShingleAnalyzerWrapper
from java.io import File


data_test = pd.read_csv(os.path.join(BASE_DIR, "data/training_set.tsv"), sep='\t')
data_val = pd.read_csv(os.path.join(BASE_DIR, "data/validation_set.tsv"), sep='\t')


def _index(name):
    """ Returns index name
    :param name:
    :return:
    """
    return os.path.join(BASE_DIR, "data/index", name)


def get_document(fname, split_by=None):
    docs = []
    _name = os.path.split(fname)[-1]
    with open(fname) as f:
        contents = f.read()
        if split_by:
            paragraphs = contents.split(split_by)
            for ix, par in enumerate(paragraphs):
                if not par:
                    continue
                doc = Document()
                name = "{}_{}".format(_name, ix)
                doc.add(Field('filename', name, Field.Store.YES, Field.Index.NOT_ANALYZED))
                doc.add(Field('content', par, Field.Store.YES, Field.Index.ANALYZED))
                docs.append(doc)
        else:
            doc = Document()
            doc.add(Field('filename', _name, Field.Store.YES, Field.Index.NOT_ANALYZED))
            doc.add(Field('content', contents, Field.Store.YES, Field.Index.ANALYZED))
            docs.append(doc)
    return docs


def is_dir(path):
    """ Checks if dir or not
    """
    if not path.startswith(BASE_DIR):
        return os.path.isdir(os.path.join(BASE_DIR, path))
    else:
        return os.path.isdir(path)


def to_document(data):
    """ Parses given data into collection of Lucene Documents
    :param dict|string data: could be either dict of documents or directory path
    :return:
    """
    docs = []
    if is_dir(data):  # parse dir
        for fname in os.listdir(data):
            doc = get_document(fname)
            docs.extend(doc)
    else:
        for fname, content in data:
            doc = Document()
            doc.add(Field('filename', fname, Field.Store.YES, Field.Index.NOT_ANALYZED))
            doc.add(Field('content', content, Field.Store.YES, Field.Index.ANALYZED))
            docs.append(doc)
    return docs


def index(analyzer, index_dest_dir, documents):
    """ Builds Lucene index from provided documents using given analyzer
    :param analyzer:
    :param index_dest_dir:
    :param list[Document] documents:
    :return:
    """
    if not all([isinstance(d, Document) for d in documents]):
        raise TypeError("documents should be iterable of type Document! Given: %s" % type(documents[0]))

    writer_config = IndexWriterConfig(Version.LUCENE_30, analyzer)
    writer = IndexWriter(SimpleFSDirectory(File(index_dest_dir)), writer_config)
    for doc in documents:
        writer.addDocument(doc)
    writer.close()


def make_request(query, analyzer, index, qparser_regexp=None, max_results=100):
    """
    :param query:
    :param analyzer:
    :param index:
    :param qparser_regexp:
    :param max_results:
    :return:
    """
    reader = IndexReader.open(SimpleFSDirectory(File(index)))
    searcher = IndexSearcher(reader)

    query = QueryParser(Version.LUCENE_30, "content", analyzer).parse(
        query if not qparser_regexp else re.sub(qparser_regexp, " ", query)
    )
    hits = searcher.search(query, max_results)
    return hits.scoreDocs


if __name__ == '__main__':
    lucene.initVM()
    q = """
    When athletes begin to exercise, their heart rates and respiration rates increase.
    At what level of organization does the human body coordinate these functions?
    at the tissue level	at the organ level
    """
    analyzer = SnowballAnalyzer(Version.LUCENE_30, "English", StandardAnalyzer.STOP_WORDS_SET)

    hits = make_request(
        query=q,
        analyzer=analyzer,
        index=_index("compound_index_all_wiki_paragraphs"),
        qparser_regexp="[^a-zA-Z0-9]",
        max_results=30
    )
    print [hit for hit in hits.scoreDocs]
