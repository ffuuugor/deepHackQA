__author__ = 'ffuuugor'
import elasticsearch
import json

q={
   "size": 100,
   "query": {
      "bool": {
         "must": {
            "match": {
              "title.english": {
                "query": "dog",
                "minimum_should_match": "50%"
              }
            }
         },
         "should": {
             "match_phrase": {
              "title.english": {
                "query": "dog",
                "slop": 50
              }
            }
         }
      }
   }
}
# q={
#   "query": {
#     "match_phrase": {
#       "title.english": {
#         "query": "quick dog",
#         "slop": 50
#         # "minimum_should_match": "50%"
#       }
#     }
#   }
# }
# The quick brown fox
# The quick brown fox jumps over the lazy dog
# The quick brown fox jumps over the quick dog
# Brown fox brown dog
es = elasticsearch.Elasticsearch(hosts=["elvis.zvq.me","kurt.zvq.me","jimmy.zvq.me"])

res = es.search(index="foxes",doc_type="wiki-type",body=q)
print json.dumps(res, indent=4)
