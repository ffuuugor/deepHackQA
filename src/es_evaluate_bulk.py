__author__ = 'ffuuugor'
import csv
import elasticsearch
import logging
from greq import bulk_es_search
from collections import namedtuple
import sys

Question = namedtuple("Question", ["id","question","correct","options"])
BATCH_SIZE=10

index = sys.argv[1]
size = int(sys.argv[2])

with open("data/validation_set.tsv") as f:
    with open("submission_es_04384.csv","w") as fout:
        reader = csv.reader(f, delimiter='\t')
        reader.next()

        correct_cnt = 0
        total_cnt = 0

        questions = []
        for line in reader:
            id = line[0]
            question_text = line[1]
            # correct_ans = line[2]
            options = line[2:]

            questions.append(Question(id=id, question=question_text, correct=None, options=options))

        for idx in range(0, len(questions), BATCH_SIZE):
            queries = []
            curr_questions = questions[idx:idx+BATCH_SIZE]
            for q in curr_questions:
                four_queries = [q.question + " " + x for x in q.options]
                queries.extend(four_queries)

            elastic_res = bulk_es_search(index,queries, size=size)

            def get_top_score(respone):
                try:
                    if len(respone["hits"]["hits"]) > 0:
                        # return respone["hits"]["hits"][0]["_score"]
                        return sum([x["_score"] for x in respone["hits"]["hits"]])
                    else:
                        return 0
                except:
                        return 0

            scores = map(get_top_score, elastic_res)

            for j in range(0, len(scores), 4):
                curr_scores = scores[j:j+4]
                curr_question = curr_questions[j/4]

                curr_scores = zip(curr_scores, ["A","B","C","D"])

                # print sorted(curr_scores, key=lambda x: x[0]), curr_question.correct
                predicted = max(curr_scores, key=lambda x: x[0])[1]
                print >> fout, "%s,%s" % (curr_question.id, predicted)
                # print predicted == curr_question.correct
                #
                # if predicted == curr_question.correct:
                #     correct_cnt += 1
                total_cnt += 1
                #
                if total_cnt%10==0:
                    print "%d ====%.4f====%s,%d" % (total_cnt, float(correct_cnt)/total_cnt, index, size)

print "========="
print float(correct_cnt)/total_cnt
