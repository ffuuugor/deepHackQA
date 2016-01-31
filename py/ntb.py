import sys
import lucene
import os
# import utils
import book_processing
from ck12_docs_generator import one_per_h1_docs
import pandas as pd
import numpy as np
import re
from lucene import *
lucene.initVM()
from java.io import File
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

data_test = pd.read_csv("data/training_set.tsv", sep = '\t')
data_val = pd.read_csv("data/validation_set.tsv", sep = '\t')

def getDocument(fname):
    doc = Document()
    doc.add(Field('filename', os.path.split(fname)[-1], Field.Store.YES, Field.Index.NOT_ANALYZED))
    doc.add(Field('content', open(fname).read(), Field.Store.YES, Field.Index.ANALYZED))
    return doc

def indexFile(fname, writer):
    doc = getDocument(fname)
    writer.addDocument(doc)

def indexDirectory(dir_path, writer):
    for fname in os.listdir(dir_path):
        indexFile(os.path.join(dir_path, fname), writer)
    return writer.numDocs()

def indexDictionary(d, writer):
    for k, v in d.iteritems():
        doc = Document()
        doc.add(Field('filename', k, Field.Store.YES, Field.Index.NOT_ANALYZED))
        doc.add(Field('content', v, Field.Store.YES, Field.Index.ANALYZED))
        writer.addDocument(doc)
    return writer.numDocs()

#index wiki articles based on ck 12 topics
#analyzer = StandardAnalyzer(Version.LUCENE_30)
# analyzer = SnowballAnalyzer(Version.LUCENE_30, "English", StandardAnalyzer.STOP_WORDS_SET)
# indexWriterConfig = IndexWriterConfig(Version.LUCENE_30, analyzer)
# # writer = IndexWriter(SimpleFSDirectory(File("data/index/wiki_ck12")), analyzer, True, LimitTokenCountAnalyzer)
# writer = IndexWriter(SimpleFSDirectory(File("data/index/wiki_ck12")), indexWriterConfig)
# indexDirectory('data/wiki_data', writer)
#
# writer.close()

#index topics from ck12 book (document is text between h1 tags)
# dir_name = 'data/ck12_book/OEBPS'
# docs = {}
# html_paths = [os.path.join(dir_name,  str(i+1) + '.html') for i in range(124)]
# for f_name in html_paths:
#     docs.update(book_processing.get_h1_text(open(f_name).read()))
#
# #analyzer = StandardAnalyzer(Version.LUCENE_30)
# analyzer = SnowballAnalyzer(Version.LUCENE_30, "English", StandardAnalyzer.STOP_WORDS_SET)
# indexWriterConfig = IndexWriterConfig(Version.LUCENE_30, analyzer)
# writer = IndexWriter(SimpleFSDirectory(File("data/index/ck12_books_topics")), indexWriterConfig)
# indexDictionary(docs, writer)
#
# writer.close()

#index paragraphs from ck12 book (document is text between any h tags)
# dir_name = 'data/ck12_book/OEBPS'
# docs = {}
# html_paths = [os.path.join(dir_name,  str(i+1) + '.html') for i in range(124)]
# for f_name in html_paths:
#     docs.update(book_processing.get_h_all_text(open(f_name).read()))
#
# #analyzer = StandardAnalyzer(Version.LUCENE_30)
# analyzer = SnowballAnalyzer(Version.LUCENE_30, "English", StandardAnalyzer.STOP_WORDS_SET)
# indexWriterConfig = IndexWriterConfig(Version.LUCENE_30, analyzer)
# writer = IndexWriter(SimpleFSDirectory(File("data/index/ck12_books_paragraphs")), indexWriterConfig)
# indexDictionary(docs, writer)
#
# writer.close()

dir_name = 'data/ck12_book/OEBPS'
docs = {}
html_paths = [os.path.join(dir_name,  str(i+1) + '.html') for i in range(124)]
#
for f_name in html_paths:
    docs.update(book_processing.get_h_all_text(open(f_name).read()))

for fname in os.listdir("data/allwiki"):
    content = open(os.path.join("data","allwiki", fname)).read()
    i = 0
    for doc in content.split("\n"):
        if len(doc.strip()) > 0:
            docs["%s%d" % (fname, i)] = doc
            i += 1

#analyzer = StandardAnalyzer(Version.LUCENE_30)
analyzer = SnowballAnalyzer(Version.LUCENE_30, "English", StandardAnalyzer.STOP_WORDS_SET)
indexWriterConfig = IndexWriterConfig(Version.LUCENE_30, analyzer)
writer = IndexWriter(SimpleFSDirectory(File("data/index/combo4")), indexWriterConfig)
indexDictionary(docs, writer)

writer.close()
