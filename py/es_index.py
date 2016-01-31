__author__ = 'ffuuugor'
import elasticsearch
import json
import os


def index_dir(dir, index="wiki-test3", type="wiki-type", bulk_size=32):
    """
    Example method to index some documents
    :param dir: directory with text files to index
    :return:
    """
    es = elasticsearch.Elasticsearch(hosts=["elvis.zvq.me","kurt.zvq.me","jimmy.zvq.me"])

    bulk_query = []
    for filename in os.listdir(dir):
        with open(os.path.join(dir,filename),"r") as f:
            content = f.read()
            for doc in content.split("\n"):
                if len(doc.strip()) > 0:
                    bulk_query.append({"index": {"_type":type}})
                    bulk_query.append({"title":doc})

    print "total=%d" % len(bulk_query)
    for x in range(0,len(bulk_query), bulk_size):
        while True:
            try:
                currq = bulk_query[x:x+bulk_size]
                print x
                es.bulk(currq, index=index)
                break
            except:
                "ERROR"

def index_docs(docs, index="wiki-test3", type="wiki-type", bulk_size=32):
    es = elasticsearch.Elasticsearch(hosts=["elvis.zvq.me","kurt.zvq.me","jimmy.zvq.me"])

    bulk_query = []
    for doc in docs:
        print len(doc)
        bulk_query.append({"index": {"_type":type}})
        bulk_query.append({"title":doc})

    for x in range(0,len(bulk_query), bulk_size):
        while True:
            try:
                currq = bulk_query[x:x+bulk_size]
                print x
                es.bulk(currq, index=index)
                break
            except:
                print "ERROR"


if __name__ == '__main__':
    index_dir("data/wiki3",index="wiki_h1a", bulk_size=256)
    # import book_processing
    # import re
    # dir_name = 'data/ck12_book/OEBPS'
    # docs = []
    # html_paths = [os.path.join(dir_name,  str(i+1) + '.html') for i in range(124)]
    #
    # for f_name in html_paths:
    #     docs.extend(book_processing.get_h_all_text(open(f_name).read()).values())
    # # docs = book_processing.get_h_all_text()
    # docs = [re.sub("\s+"," ",x) for x in docs]
    # index_docs(docs,index="ck12_h2_snowball")
