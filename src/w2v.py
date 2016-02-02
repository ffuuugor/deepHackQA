# coding: utf-8
"""
created by artemkorkhov at 2016/02/02
"""

import os
import json
from pprint import pprint
import itertools
import collections

import numpy as np
import pandas as pd

from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from gensim.utils import tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from spacy import English

from src.utils.text import get_questions as gq
from src import BASE_DIR


STOPS = set(stopwords.words("english"))

WORD2VEC = {}
with open("data/glove/glove.6B." + str(300) + "d.txt") as f:
    for line in f:
        l = line.split()
        WORD2VEC[l[0]] = map(float, l[1:])


def _preprocess_text(text, stem=False):
    """ Performs common atomic operations on one text chunk - tokenization, normalization
    :param text:
    :return:
    """
    words = filter(lambda x: x not in STOPS, map(lambda x: x.lower(), tokenize(text)))
    if stem:
        porter = PorterStemmer()
        words = map(porter.stem, words)
    return words


def _sum_vec(vecs):
    """
    :param vecs:
    :return:
    """
    sum_ = np.zeros(300)
    for v in vecs:
        sum_ += v
    sum_ /= np.linalg.norm(sum_)
    return sum_


def _load_json_list(name):
    """
    :param qtype:
    :return:
    """
    if os.path.exists(os.path.join(BASE_DIR, name)):
        with open(os.path.join(BASE_DIR, name)) as f:
            return json.loads(f.read())
    else:
        return None


def _save_json_list(list_, name):
    """
    :param qtype:
    :return:
    """
    with open(os.path.join(BASE_DIR, name), 'wb') as f:
        json.dump(list_, f)


def get_tfidf_model(path="data/swiki.json", save_path="data/swiki_dict.txt", stem=False):
    """
    :param path:
    :param save_path:
    :return:
    """
    texts = map(lambda x: _preprocess_text(x, stem=stem), _load_json_list("data/swiki.json"))

    def _get_swiki_dictionary():
        dict_file = os.path.join(BASE_DIR, save_path)
        if os.path.exists(dict_file):
            dictionary = corpora.Dictionary.load_from_text(dict_file)
        else:
            dictionary = corpora.Dictionary(texts)
            dictionary.save_as_text(dict_file)
        return dictionary

    dct = _get_swiki_dictionary()

    bow_texts = map(dct.doc2bow, texts)
    tfidf = TfidfModel(bow_texts)
    return dct, tfidf


def get_top_keywords(dct, tfidf, top=0.5):
    """
    :param dct:
    :param tfidf:
    :return:
    """
    toplist = []
    num_docs = len(tfidf.idfs)
    for i in range(num_docs):
        if tfidf.dfs.get(i) and tfidf.idfs.get(i):
            toplist.append((i, (tfidf.dfs[i] * 1. / num_docs) * tfidf.idfs[i]))

    ordered = sorted(toplist, key=lambda x: -x[1])
    triples = []
    for el in ordered:
        if dct.get(el[0]):
            triples.append((el[0], el[1], dct[el[0]]))

    return triples[:int(len(triples)*top)]


def get_questions(qtype='TS', keywords=None):
    """ Returns normalized question pairs
    :param qtype:
    :param keywords:
    :return:
    """
    questions = _load_json_list("data/cleaned_questions_%s.json" % qtype)
    if questions is not None:
        return questions
    else:
        questions = gq(qtype)
        processed_qs = []
        for qas in questions:
            word_lists = map(_preprocess_text, qas)
            if keywords:
                packs = []
                for word_pack in word_lists:  # iterate over qestion/answer candidates
                    filtered = filter(lambda x: x in keywords, word_pack)
                    packs.append(filtered)
                processed_qs.append(packs)
            else:
                processed_qs.append(word_lists)
        _save_json_list(processed_qs, "data/cleaned_questions_%s.json" % qtype)
        return processed_qs


def to_vectorspace(itemlist):
    """
    :param itemlist:
    :return:
    """
    vectorspace = []
    initial_len = len(itemlist)
    final_len = 0
    for word in itemlist:
        if word in WORD2VEC:
            vectorspace.append(WORD2VEC[word])
            final_len += 1
    return vectorspace, initial_len, final_len


def permute(vectorset):
    """ performs all to all permutations for +- of every vector in given vectorset
    :param vectorset:
    :return:
    """
    import ipdb
    ipdb.set_trace()
    letters = 'abcdefghijklmnoprstq'

    def mask(vectorset):
        _mask = {}
        for ix, vector in enumerate(vectorset):
            _mask[letters[ix]] = vector
        return _mask

    def extend_mask(mask):
        result = {}
        result.update(mask)
        for el, vector in mask.items():
            result.update({str('-'+str(el)): np.dot(vector, -1.)})
        return result

    def cfilter(comb):
        options = {l: 0 for l in letters[:len(comb)]}
        for c in comb:
            if len(c) == 1:
                options[c] += 1
            if len(c) == 2:
                options[c[1]] += 1
        if 2 in options.values():
            return False
        else:
            return True

    masked = mask(vectorset)
    extended_mask = extend_mask(masked)  # (m, vector) pairs
    permutations = itertools.combinations(extended_mask.keys(), len(masked))  # makes all combinations
    good_permutations = filter(cfilter, permutations)

    results = []
    for permutation in good_permutations:
        vec = np.zeros(300)
        for i in permutation:
            vec += extended_mask[i]

        vec /= np.linalg.norm(vec)
        results.append(vec)
    return results


def best_each_strategy(question_candidates, answer_options):
    """
    :param question_candidates:
    :param answer_options:
    :return:
    """
    aA = []
    aB = []
    aC = []
    aD = []
    answer_matrix = [aA, aB, aC, aD]
    for candidate in question_candidates:
        for ix, answer in enumerate(answer_options):
            answer_matrix[ix].append(np.dot(candidate, answer))
    answer = np.argmax(map(lambda x: sorted(x)[-1], answer_matrix))
    return 'ABCD'[answer]


def best_overall_strategy(question_candidates, answer_options):
    """
    :param question_candidates:
    :param answer_options:
    :return:
    """
    competitors = []
    for candidate in question_candidates:
        answers = []
        for ix, answer in enumerate(answer_options):
            answers.append((ix, np.dot(candidate, answer)))
        best = sorted(answers, key=lambda x: x[-1])[-1]
        competitors.append(best)
    best_answer = sorted(competitors, lambda x: x[-1])[-1]
    return 'ABCD'[best_answer[0]]


def main(qtype='TS'):
    """
    :return:
    """
    # TODO 1 ST ITERATION - permutations only on questions
    dictionary, tfidf = get_tfidf_model()
    top_kw_triples = get_top_keywords(dictionary, tfidf, top=0.75)

    keywords = [el[2] for el in top_kw_triples]
    qa_wordlists = get_questions(qtype=qtype, keywords=keywords)

    import ipdb
    ipdb.set_trace()

    questions_stats = collections.Counter()
    answers_stats = collections.Counter()

    results = _load_json_list("data/qa_candidate_vectors_%s.json" % qtype)
    if results is None:
        results = []
        for qas in qa_wordlists:
            question, ans_a, ans_b, ans_c, ans_d = qas
            qvecspace, q_init_length, q_final_length = to_vectorspace(question)
            questions_stats.update([(q_final_length * 1.) / q_init_length])
            question_candidates_vectors = permute(qvecspace)
            answer_candidates = []
            for ac in [ans_a, ans_b, ans_c, ans_d]:
                vecs, a_init_len, a_final_len = to_vectorspace(ac)
                answers_stats.update([(a_final_len * 1.)/a_init_len])
                answer_candidate = _sum_vec(vecs)
                answer_candidates.append(answer_candidate)
            results.append((question_candidates_vectors, answer_candidates))
        _save_json_list(results, "data/qa_candidate_vectors_%s.json" % qtype)

    best_each_answers = {}
    best_overall_answers = {}

    for ix, (question_candidates, answer_vectors) in enumerate(results):
        best_each_answers.update({ix+1, best_each_strategy(question_candidates, answer_vectors)})
        best_overall_answers.update({ix+1, best_overall_strategy(question_candidates, answer_vectors)})

    best_each = pd.DataFrame(data=best_each_answers)
    best_each.to_csv(
        os.path.join(BASE_DIR, "data/best_each_answers_strategy_results.csv"),
        index=False, sep='\t'
    )
    best_overall = pd.DataFrame(data=best_overall_answers)
    best_overall.to_csv(
        os.path.join(BASE_DIR, "data/best_each_answers_strategy_results.csv"),
        index=False, sep='\t'
    )

    TS = pd.read_csv(os.path.join(BASE_DIR, "data/training_set.tsv"), sep='\t')


if __name__ == '__main__':

    main()
