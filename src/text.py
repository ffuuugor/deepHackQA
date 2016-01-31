# coding: utf-8
"""
created by artemkorkhov at 2016/01/20
"""

import nltk


sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


def qa_pair(question, answer_option):
    """ Joins question text with answer option.
    :param question:
    :param answer_option:
    :return:
    """
    space_left = True
    space_right = True
    if '_' in question:
        l, r = question.find("_"), question.rfind("_")
        if l > 0 and question[l-1] != " ":
            space_left = False
        if l < len(question) - 2 and question[r+1] != " ":
            space_right = False

        substr = "{sl}{txt}{sr}".format(
            sl="" if space_left else " ",
            txt=answer_option,
            sr="" if space_right else " "
        )
        q = question[:l] + substr + question[r+1:]
    else:
        q = question + " " + answer_option
    return q


def tokenize_sentences(document):
    """ Splits given doc by sentences
    :param document:
    :return:
    """
    return sentence_tokenizer.tokenize(document)


# def tokenize(doc,
#              pattern=r'''(?x)    # set flag to allow verbose regexps
#                  ([A-Z]\.)+        # abbreviations, e.g. U.S.A.
#                | \w+(-\w+)*        # words with optional internal hyphens
#                | \$?\d+(\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
#                | \.\.\.            # ellipsis
#                | [][.,;"'?():-_`]  # these are separate tokens; includes ], [
#             '''):
#     """ Tokenize doc. Defines tokens by given pattern
#     :param doc:
#     :return:
#     """
#     return nltk.regexp_tokenize(doc, pattern)

