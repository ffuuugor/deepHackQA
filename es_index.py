__author__ = 'ffuuugor'
import elasticsearch
import json
import os


def index_docs(dir, index="wiki-test3", type="wiki-type", bulk_size=32):
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
            print len(content)

            bulk_query.append({"index": {"_type":type}})
            bulk_query.append({"title":content})

    for x in range(0,len(bulk_query), bulk_size):
        currq = bulk_query[x:x+bulk_size]
        print x
        es.bulk(currq, index=index)

if __name__ == '__main__':
    index_docs("wiki_data")

