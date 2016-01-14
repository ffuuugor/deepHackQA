__author__ = 'ffuuugor'
import csv
import elasticsearch
import logging

es = elasticsearch.Elasticsearch(hosts=["elvis.zvq.me","kurt.zvq.me","jimmy.zvq.me"])

with open("training_set.tsv") as f:
    reader = csv.reader(f, delimiter='\t')
    reader.next()

    correct_cnt = 0
    total_cnt = 0
    for line in reader:
        id = line[0]
        question = line[1]
        correct = line[2]
        options = line[3:]

        scores = []
        for idx, opt in enumerate(["A","B","C","D"]):
            try:
                q = question + " " + options[idx]

                query = {
                   "query": {
                      "bool": {
                         "must": {
                            "multi_match": {
                                "type": "most_fields",
                                "query": q,
                                "fields": [ "title", "title.english" ],
                                "minimum_should_match": "30%"
                            }
                         },
                         "should": {
                             "multi_match": {
                                "type": "most_fields",
                                "query": q,
                                "fields": [ "title", "title.english", "title.shingles" ]
                            }
                         }
                      }
                   }
                }

                elastic_res = es.search(index="wiki-test3",doc_type="wiki-type",body=query)
                if len(elastic_res["hits"]["hits"]) > 0:
                    score = elastic_res["hits"]["hits"][0]["_score"]
                    scores.append((opt, score))
            except:
                scores.append((opt,0))

        print sorted(scores, key=lambda x: x[1]), correct
        predicted = max(scores, key=lambda x: x[1])[0]
        print predicted == correct

        if predicted == correct:
            correct_cnt += 1
        total_cnt += 1

        if total_cnt%10==0:
            print "====%.4f====" % (float(correct_cnt)/total_cnt)

print "========="
print float(correct_cnt)/total_cnt
