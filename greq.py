__author__ = 'ffuuugor'
import grequests
import json
import random

def _construct_query(query_string):
    return {
            "query": {
                "bool": {
                    "must": {
                        "multi_match": {
                                "type": "most_fields",
                                "query": query_string,
                                "fields": [ "title", "title.english" ],
                                "minimum_should_match": "30%"
                            }
                    },
                    "should": {
                            "multi_match": {
                                "type": "most_fields",
                                "query": query_string,
                                "fields": [ "title", "title.english", "title.shingles" ]
                            }
                        }
                    }
                }
            }

def _construct_url(host, index):
    """
    Construct url to perform search api call to ES
    :param host:
    :param index:
    :return:
    """
    return "http://%s:9200/%s/_search" % (host, index)

def bulk_es_search(index, queries, hosts=None, batch_size=32):
    """
    Perform bulk search request to elasticsearch
    :param index: index name
    :param queries: list of query_strings
    :param hosts: elastic hosts
    :param batch_size:
    :return: list of elastic responses with respect to original order
    """
    if hosts is None:
        hosts=["elvis.zvq.me","kurt.zvq.me","jimmy.zvq.me"]

    ret = []
    for idx in range(0, len(queries), batch_size):

        rs = (grequests.post(_construct_url(hosts[int(random.random()*len(hosts))], index),
                             data=json.dumps(_construct_query(q))) for q in queries[idx:idx+batch_size])
        ret.extend(map(lambda x: json.loads(x.text), grequests.map(rs)))

    return ret

if __name__ == '__main__':
    print json.dumps(bulk_es_search("wiki-test3", queries=['fox','brown']), indent=4)
