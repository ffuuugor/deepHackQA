# coding: utf-8
"""
created by artemkorkhov at 2016/02/01
"""

import os
import logging

import numpy as np
from lda2vec import preprocess, Corpus, LDA2Vec
from chainer import serializers

from src.utils.text import get_docs, make_texts, get_questions, preprocess_text


log = logging.getLogger(__name__)


def get_tokens():
    """
    :return:
    """
    docs = get_docs()
    texts = make_texts(docs, single=False)
    questions = get_questions()

    texts.extend(questions)

    texts = preprocess_text(texts)
    texts = [t for t in texts if t]

    tokens, vocab = preprocess.tokenize(texts, 7500, tag=False, parse=False, entity=False)
    return tokens, vocab


# todo figure ot proper min_count. Need to see the the real distribution
def make_corpus(tokens, min_count=50):
    """ Creates LDA2vec corpus
    :param text:
    :return:
    """
    corpus = Corpus()
    corpus.update_word_count(tokens)
    corpus.finalize()

    compact = corpus.to_compact(tokens)

    pruned = corpus.filter_count(compact, min_count=min_count)
    clean = corpus.subsample_frequent(pruned)
    doc_ids = np.arange(pruned.shape[0])
    corpus, flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)
    return corpus, flattened, doc_ids, clean


def main():

    docs = get_docs()
    texts = make_texts(docs, single=False)
    questions = get_questions()
    texts.extend(questions)
    texts = preprocess_text(texts)
    texts = [t for t in texts if t]

    tokens, vocab = preprocess.tokenize(texts, 7500, tag=False, parse=False, entity=False)
    log.info("Got tokens and vocabulary. Vocab size: %d" % len(vocab))

    corpus, flat_corpus, doc_ids, clean_set = make_corpus(tokens=tokens, min_count=50)
    log.info("Got corpus")

    # Model Parameters
    # Number of documents
    n_docs = len(texts)
    log.info("number of texts: %d" % n_docs)
    # Number of unique words in the vocabulary
    n_words = flat_corpus.max() + 1
    # Number of dimensions in a single word vector
    n_hidden = 128
    # Number of topics to fit
    n_topics = 20
    # Get the count for each key
    counts = corpus.keys_counts[:n_words]
    # Get the string representation for every compact key
    words = corpus.word_list(vocab)[:n_words]
    log.info("Words: \n %s" % words)

    # Fit the model
    log.info("fitting the model")
    model = LDA2Vec(n_words, n_hidden, counts, dropout_ratio=0.2)
    model.add_categorical_feature(n_docs, n_topics, name='document_id')
    model.finalize()
    if os.path.exists('model.hdf5'):
        serializers.load_hdf5('model.hdf5', model)
    for _ in range(200):
        log.info("attempt #%d" % _)
        model.top_words_per_topic('document_id', words)
        log.info("TOP_WORDS_PER_TOPIC!\n => ")
        log.info(model.top_words_per_topic('document_id', words))
        log.info('========')
        model.fit(flat_corpus, categorical_features=[doc_ids], fraction=1e-3,
                  epochs=1)
        model.to_cpu()
    serializers.save_hdf5('model.hdf5', model)
    model.top_words_per_topic('document_id', words)


if __name__ == '__main__':
    main()




