# coding: utf-8
"""
created by artemkorkhov at 2016/01/17
"""
 
from pprint import pprint
import sys
import lucene
import os
# import utils
import book_processing
import pandas as pd
import numpy as np
import re
from lucene import *
import pandas
 
lucene.initVM()
 
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
from java.io import File
 
 
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
 
 
def make_index(indexed_data, index_destination, source='directory'):
    #index wiki articles based on ck 12 topics
    #analyzer = StandardAnalyzer(Version.LUCENE_30)
    analyzer = SnowballAnalyzer(Version.LUCENE_30, "English", StandardAnalyzer.STOP_WORDS_SET)
    indexWriterConfig = IndexWriterConfig(Version.LUCENE_30, analyzer)
    writer = IndexWriter(SimpleFSDirectory(File(index_destination)), indexWriterConfig)
    if source == 'directory':
        indexDirectory(indexed_data, writer)
    else:
        indexDictionary(indexed_data, writer)
    writer.close()
 
 
def predict_test(indexed_data, index_destination, source='directory', already_indexed=False):
    """
    :param indexed_data_dir:
    :param index_destination:
    :return:
    """
    def choose_best():
        scores = []
        for k, v in sorted(res.items(), key=lambda x: x[0]):
            scores.append((k, 1. * sum(data_test['correctAnswer'] == v) / len(v)))
        return sorted(scores, key=lambda x: -x[-1])[0][0]
 
    def calculate_score(res):
        """
        :param res:
        :return:
        """
        correct = 0
        total = 0
        for index, row in data_test.iterrows():
            if res[index] == row['correctAnswer']:
                correct += 1
            total += 1
        return float(correct)/total
 
    if not already_indexed:
        make_index(indexed_data, index_destination, source)
 
    res = {}
    MAX = 100
    docs_per_q = range(1,20)

    records = []
 
    #analyzer = StandardAnalyzer(Version.LUCENE_30)
    analyzer = SnowballAnalyzer(Version.LUCENE_30, "English", StandardAnalyzer.STOP_WORDS_SET)
    reader = IndexReader.open(SimpleFSDirectory(File(index_destination)))
    searcher = IndexSearcher(reader)
 
    for index, row in data_test.iterrows():
 
        queries = [row['answerA'], row['answerB'], row['answerC'], row['answerD']]
        queries = [row['question'] + ' ' + q for q in queries]
 
        scores = {}
        for q in queries:
            query = QueryParser(Version.LUCENE_30, "content", analyzer).parse(re.sub("[^a-zA-Z0-9]"," ", q))
            #query = QueryParser(Version.LUCENE_30, "content", analyzer).parse(re.sub("[/^]", "\^", q))
            hits = searcher.search(query, MAX)
            doc_importance = [hit.score for hit in hits.scoreDocs]
            for n in docs_per_q:
                scores.setdefault(n, [])
                scores[n].append(sum(doc_importance[:n]))

        to_records = [index+102501]
        to_records.append(['A','B','C','D'][np.argmax(scores[4])])
        records.append(to_records)

        for n in docs_per_q:
            res.setdefault(n, [])
            res[n].append(['A','B','C','D'][np.argmax(scores[n])])

    df = pandas.DataFrame.from_records(records, columns=["id","correctAnswer"])
    df = df.set_index("id")
    df.to_csv("ololo.csv")

    # print res[4]
    best = choose_best()
    print best
    score = calculate_score(res[best])
    # score = calculate_score(res)
    print score
 
 
if __name__ == "__main__":
    lucene.initVM()
    # predict_test(indexed_data="data/wiki_data", index_destination="data/index/wiki_ck12", already_indexed=True)
    predict_test(indexed_data="data/ck12_book/OEBPS",
                 index_destination="data/index/combo4",
                 already_indexed=True)
