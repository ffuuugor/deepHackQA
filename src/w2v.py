# coding: utf-8
"""
created by artemkorkhov at 2016/02/02
"""

import os
from pprint import pprint
import itertools

import numpy as np

from gensim import corpora
from gensim.models.tfidfmodel import TfidfModel
from gensim.utils import tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# from spacy import English

from src.utils.text import get_swiki, qa_pair
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


def get_tfidf_model(path="data/swiki.json", save_path="data/swiki_dict.txt", stem=False):
    """
    :param path:
    :param save_path:
    :return:
    """
    texts = map(lambda x: _preprocess_text(x, stem=stem), get_swiki())

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


def get_questions(keywords=None):
    """ Returns normalized question pairs
    :return:
    """
    questions = gq()
    import ipdb
    ipdb.set_trace()
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
    letters = 'abcdefghijklmnoprs'

    def mask(vectorset):
        _mask = {}
        for ix, vector in enumerate(vectorset):
            _mask[letters[ix]] = vector
        return _mask

    def extend_mask(mask):
        result = {}
        result.update(mask)
        for el, vector in mask:
            result.update({str('-'+el): vector * -1})
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
    permutations = list(itertools.combinations(extended_mask.keys(), len(masked)))  # makes all combinations
    good_permutations = filter(cfilter, permutations)

    results = []
    for permutation in good_permutations:
        vec = np.zeros(300)
        for i in permutation:
            vec += extended_mask[i]

        vec /= np.linalg.norm(vec)
        results.append(vec)
    return results


def main():
    """
    :return:
    """
    # TODO 1 ST ITERATION - permutations only on questions
    dictionary, tfidf = get_tfidf_model()
    top_kw_triples = get_top_keywords(dictionary, tfidf, top=0.75)

    keywords = [el[2] for el in top_kw_triples]
    qa_wordlists = get_questions(keywords)
    results = []
    for qas in qa_wordlists:
        q, a, b, c, d = qas
        qvecspace, q_init_length, q_final_length = to_vectorspace(q)
        question_candidates_vectors = permute(qvecspace)
        answer_candidates = []
        for ac in [a, b, c, d]:
            vecs, a_init_len, q_final_len = to_vectorspace(ac)
            answer_candidate = _sum_vec(vecs)
            answer_candidates.append(answer_candidate)
        results.append((question_candidates_vectors, answer_candidates))


if __name__ == '__main__':

    # dct, tfidf = get_tfidf_model()
    # pprint(tfidf.idfs.items()[:20])
    # pprint(tfidf.dfs.items()[:20])

    # top_kws = get_top_keywords(dct, tfidf)

    # pprint(top_kws[:20])
    questions = get_questions()
    pprint(questions[:5])

