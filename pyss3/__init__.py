# -*- coding: utf-8 -*-
"""
This is the main module containing the implementation of the SS3 classifier.

(Please, visit https://github.com/sergioburdisso/pyss3 for more info)
"""
from __future__ import print_function
import os
import re
import json
import errno

from io import open
from time import time
from tqdm import tqdm
from math import pow, tanh
from .util import Print, VERBOSITY, Preproc as Pp

# python 2 and 3 compatibility
from functools import reduce
from six.moves import xrange

__version__ = "0.4.0"

ENCODING = "utf-8"

PARA_DELTR = "\n"
SENT_DELTR = r"\."
WORD_DELTR = r"\s"

STR_UNKNOWN, STR_MOST_PROBABLE = "unknown", "most-probable"
STR_UNKNOWN_CATEGORY = "[unknown]"
IDX_UNKNOWN_CATEGORY = -1
STR_UNKNOWN_WORD = ''
IDX_UNKNOWN_WORD = -1
STR_VANILLA, STR_XAI = "vanilla", "xai"
STR_GV, STR_NORM_GV, STR_NORM_GV_XAI = "gv", "norm_gv", "norm_gv_xai"

STR_MODEL_FOLDER = "ss3_models"
STR_MODEL_EXT = "ss3m"

VERBOSITY = VERBOSITY  # to allow "from pyss3 import VERBOSITY"

NAME = 0
VOCAB = 1

NEXT = 0
FR = 1
CV = 2
SG = 3
GV = 4
LV = 5
EMPTY_WORD_INFO = [0, 0, 0, 0, 0, 0]

NOISE_FR = 1
MIN_MAD_SD = .03


class SS3:
    """
    The SS3 classifier class.

    The SS3 classifier was originally  defined in Section 3 of
    https://dx.doi.org/10.1016/j.eswa.2019.05.023
    (preprint avialable here: https://arxiv.org/abs/1905.08772)

    :param s: the "smoothness"(sigma) hyperparameter value
    :type s: float
    :param l: the "significance"(lambda) hyperparameter value
    :type l: float
    :param p: the "sanction"(rho) hyperparameter value
    :type p: float
    :param a: the alpha hyperparameter value (i.e. all terms with a
              confidence value (cv) less than alpha will be ignored during
              classification)
    :type a: float
    :param name: the model's name (to save and load the model from disk)
    :type name: str
    :param cv_m: method used to compute the confidence value (cv) of each
                 term (word or n-grams), options are:
                 "norm_gv_xai", "norm_gv" and "gv" (default: "norm_gv_xai")
    :type cv_m: str
    :param sn_m: method used to compute the sanction (sn) function, options
                 are: "vanilla" and "xai" (default: "xai")
    :type sn_m: str
    """

    __name__ = "model"
    __models_folder__ = STR_MODEL_FOLDER

    __s__ = .45
    __l__ = .5
    __p__ = 1
    __a__ = .0

    __l_update__ = None
    __s_update__ = None
    __p_update__ = None

    __prun_floor__ = 10
    __prun_trigger__ = 1000000
    __prun_counter__ = 0

    __zero_cv__ = None

    __parag_delimiter__ = PARA_DELTR
    __sent_delimiter__ = SENT_DELTR
    __word_delimiter__ = WORD_DELTR

    def __init__(
        self, s=None, l=None, p=None, a=None,
        name="", cv_m=STR_NORM_GV_XAI, sn_m=STR_XAI
    ):
        """
        Class constructor.

        :param s: the "smoothness"(sigma) hyperparameter value
        :type s: float
        :param l: the "significance"(lambda) hyperparameter value
        :type l: float
        :param p: the "sanction"(rho) hyperparameter value
        :type p: float
        :param a: the alpha hyperparameter value (i.e. all terms with a
                  confidence value (cv) less than alpha will be ignored during
                  classification)
        :type a: float
        :param name: the model's name (to save and load the model from disk)
        :type name: str
        :param cv_m: method used to compute the confidence value (cv) of each
                     term (word or n-grams), options are:
                     "norm_gv_xai", "norm_gv" and "gv" (default: "norm_gv_xai")
        :type cv_m: str
        :param sn_m: method used to compute the sanction (sn) function, options
                     are: "vanilla" and "xai" (default: "xai")
        :type sn_m: str
        """
        self.__name__ = (name or self.__name__).lower()

        self.__s__ = s or self.__s__
        self.__l__ = l or self.__l__
        self.__p__ = p or self.__p__
        self.__a__ = a or self.__a__

        self.__categories_index__ = {}
        self.__categories__ = []
        self.__max_fr__ = []
        self.__max_gv__ = []

        self.__index_to_word__ = {}
        self.__word_to_index__ = {}

        if cv_m == STR_NORM_GV_XAI:
            self.__cv__ = self.__cv_norm_gv_xai__
        elif cv_m == STR_NORM_GV:
            self.__cv__ = self.__cv_norm_gv__
        elif cv_m == STR_GV:
            self.__cv__ = self.__gv__

        if sn_m == STR_XAI:
            self.__sg__ = self.__sg_xai__
        elif sn_m == STR_VANILLA:
            self.__sg__ = self.__sg_vanilla__

        self.__cv_mode__ = cv_m
        self.__sn_mode__ = sn_m

    def __lv__(self, ngram, icat, cache=False):
        """Local value function."""
        if cache:
            return self.__trie_node__(ngram, icat)[LV]
        else:
            try:
                ilength = len(ngram) - 1
                fr = self.__trie_node__(ngram, icat)[FR]
                if fr > NOISE_FR:
                    max_fr = self.__max_fr__[icat][ilength]
                    local_value = (fr / float(max_fr)) ** self.__s__
                    return local_value
                else:
                    return 0
            except TypeError:
                return 0
            except IndexError:
                return 0

    def __sn__(self, ngram, icat):
        """The sanction (sn) function."""
        m_values = [
            self.__sg__(ngram, ic)
            for ic in xrange(len(self.__categories__)) if ic != icat
        ]

        c = len(self.__categories__)

        s = sum([min(v, 1) for v in m_values])

        return pow((c - (s + 1)) / ((c - 1) * (s + 1)), self.__p__)

    def __sg_vanilla__(self, ngram, icat, cache=True):
        """The original significance (sg) function definition."""
        try:
            if cache:
                return self.__trie_node__(ngram, icat)[SG]
            else:
                ncats = len(self.__categories__)
                l = self.__l__
                lvs = [self.__lv__(ngram, ic) for ic in xrange(ncats)]
                lv = lvs[icat]

                M, sd = mad(lvs, ncats)

                if not sd and lv:
                    return 1.
                else:
                    return sigmoid(lv - M, l * sd)
        except TypeError:
            return 0.

    def __sg_xai__(self, ngram, icat, cache=True):
        """
        A variation of the significance (sn) function.

        This version of the sg function adds extra checks to
        improve visual explanations.
        """
        try:
            if cache:
                return self.__trie_node__(ngram, icat)[SG]
            else:
                ncats = len(self.__categories__)
                l = self.__l__

                lvs = [self.__lv__(ngram, ic) for ic in xrange(ncats)]
                lv = lvs[icat]

                M, sd = mad(lvs, ncats)

                if l * sd <= MIN_MAD_SD:
                    sd = MIN_MAD_SD / l if l else 0

                # stopwords filter
                stopword = (M > .2) or (
                    sum(map(lambda v: v > 0.09, lvs)) == ncats
                )
                if (stopword and sd <= .1) or (M >= .3):
                    return 0.

                if not sd and lv:
                    return 1.

                return sigmoid(lv - M, l * sd)
        except TypeError:
            return 0.

    def __gv__(self, ngram, icat, cache=True):
        """
        The global value (gv) function.

        This is the original way of computing the confidence value (cv)
        of a term.
        """
        if cache:
            return self.__trie_node__(ngram, icat)[GV]
        else:
            lv = self.__lv__(ngram, icat)
            weight = self.__sg__(ngram, icat) * self.__sn__(ngram, icat)
            return lv * weight

    def __cv_norm_gv__(self, ngram, icat, cache=True):
        """
        Alternative way of computing the confidence value (cv) of terms.

        This variations normalizes the gv value and uses that value as the cv.
        """
        try:
            if cache:
                return self.__trie_node__(ngram, icat)[CV]
            else:
                try:
                    cv = self.__gv__(ngram, icat)
                    return cv / self.__max_gv__[icat][len(ngram) - 1]
                except (ZeroDivisionError, IndexError):
                    return .0

        except TypeError:
            return 0

    def __cv_norm_gv_xai__(self, ngram, icat, cache=True):
        """
        Alternative way of computing the confidence value (cv) of terms.

        This variations not only normalizes the gv value but also adds extra
        checks to improve visual explanations.
        """
        try:
            if cache:
                return self.__trie_node__(ngram, icat)[CV]
            else:
                try:
                    max_gv = self.__max_gv__[icat][len(ngram) - 1]
                    if (len(ngram) > 1):
                        # stopwords guard
                        n_cats = len(self.__categories__)
                        cats = xrange(n_cats)
                        sum_words_gv = sum([
                            self.__gv__([w], ic) for w in ngram for ic in cats
                        ])
                        if (sum_words_gv < .05):
                            return .0
                        elif len([
                            w for w in ngram
                            if self.__gv__([w], icat) >= .01
                        ]) == len(ngram):
                            gv = self.__gv__(ngram, icat)
                            return gv / max_gv + sum_words_gv
                            # return gv / max_gv * len(ngram)

                    gv = self.__gv__(ngram, icat)
                    return gv / max_gv
                except (ZeroDivisionError, IndexError):
                    return .0

        except TypeError:
            return 0

    def __classify_ngram__(self, ngram):
        """Classify the given n-gram."""
        cv = [
            self.__cv__(ngram, icat)
            for icat in xrange(len(self.__categories__))
        ]
        cv[:] = [(v if v > self.__a__ else 0) for v in cv]
        return cv

    def __classify_sentence__(self, sent, prep, json=False):
        """Classify the given sentence."""
        classify_trans = self.__classify_ngram__
        cats = xrange(len(self.__categories__))
        word_index = self.get_word_index
        word_delimiter = self.__word_delimiter__
        if not json:
            if prep:
                sent = Pp.clean_and_ready(sent)
            sent_words = [
                (w, w)
                for w in re.split(word_delimiter, sent)
                if w
            ]
        else:
            if prep:
                sent_words = [
                    (w, Pp.clean_and_ready(w, dots=False))
                    for w in re_split_keep(word_delimiter, sent)
                    if w
                ]
            else:
                sent_words = [
                    (w, w)
                    for w in re_split_keep(word_delimiter, sent)
                    if w
                ]

        if not sent_words:
            sent_words = [(u'.', u'.')]

        flat_sent = []
        flat_raw_sent = []
        for raw_seq, seq in sent_words:
            words = re.split(word_delimiter, seq)
            for iw in xrange(len(words)):
                word = words[iw]

                if json and re.match(self.__word_delimiter__, raw_seq):
                    if len(flat_raw_sent):
                        flat_raw_sent[-1] += raw_seq
                        continue

                wordi = word_index(word)

                if iw == len(words) - 1:
                    word_iend = len(raw_seq)
                else:
                    if not word.startswith("nnbrr"):
                        try:
                            word_iend = re.search(word, raw_seq, re.I).end()
                        except AttributeError:
                            word_iend = len(word)
                    else:
                        word_iend = re.search(r"\d+", raw_seq).end()

                flat_sent.append(wordi)
                flat_raw_sent.append(raw_seq[:word_iend])
                raw_seq = raw_seq[word_iend:]

        sent = []
        fs_len = len(flat_sent)
        wcur = 0
        while wcur < fs_len:

            cats_ngrams = [[] for icat in cats]
            cats_max = [.0 for icat in cats]
            for icat in cats:
                woffset = 0
                word = flat_sent[wcur]
                word_info = self.__categories__[icat][VOCAB]
                while word in word_info:
                    cats_ngrams[icat].append(word_info[word][CV])
                    word_info = word_info[word][NEXT]
                    woffset += 1
                    if wcur + woffset >= fs_len:
                        break
                    word = flat_sent[wcur + woffset]

                cats_max[icat] = (max(cats_ngrams[icat])
                                  if cats_ngrams[icat] else .0)

            max_gv = max(cats_max)
            if (max_gv > self.__a__):
                cat_max_gv = cats_max.index(max_gv)
                ngram_len = cats_ngrams[cat_max_gv].index(max_gv) + 1

                max_gv_sum_1_grams = max([
                    sum([
                        self.__categories__[ic][VOCAB][w][CV]
                        if w in self.__categories__[ic][VOCAB]
                        else 0
                        for w
                        in flat_sent[wcur:wcur + ngram_len]
                    ])
                    for ic in cats
                ])

                if (max_gv_sum_1_grams > max_gv):
                    ngram_len = 1
            else:
                ngram_len = 1

            sent.append(
                (
                    flat_raw_sent[wcur:wcur + ngram_len],
                    flat_sent[wcur:wcur + ngram_len]
                )
            )
            wcur += ngram_len

        get_word = self.get_word
        if not json:
            words_cvs = [classify_trans(seq) for _, seq in sent]
            if words_cvs:
                return self.summary_op_ngrams(words_cvs)
            return self.__zero_cv__
        else:
            get_tip = self.__trie_node__
            local_value = self.__lv__
            info = [
                {
                    "token": u"→".join(map(get_word, sequence)),
                    "lexeme": u"".join(raw_sequence),
                    "cv": classify_trans(sequence),
                    "lv": [local_value(sequence, ic) for ic in cats],
                    "fr": [get_tip(sequence, ic)[FR] for ic in cats]
                }
                for raw_sequence, sequence in sent
            ]
            return {
                "words": info,
                "cv": self.summary_op_ngrams([v["cv"] for v in info]),
                "wmv": reduce(vmax, [v["cv"] for v in info])  # word max value
            }

    def __classify_paragraph__(self, parag, prep, json=False):
        """Classify the given paragraph."""
        if not json:
            sents_cvs = [
                self.__classify_sentence__(sent, prep=prep)
                for sent in re.split(self.__sent_delimiter__, parag)
                if sent
            ]
            if sents_cvs:
                return self.summary_op_sentences(sents_cvs)
            return self.__zero_cv__
        else:
            info = [
                self.__classify_sentence__(sent, prep=prep, json=True)
                for sent in re_split_keep(self.__sent_delimiter__, parag)
                if sent
            ]
            if info:
                sents_cvs = [v["cv"] for v in info]
                cv = self.summary_op_sentences(sents_cvs)
                wmv = reduce(vmax, [v["wmv"] for v in info])
            else:
                cv = self.__zero_cv__
                wmv = cv
            return {
                "sents": info,
                "cv": cv,
                "wmv": wmv  # word max value
            }

    def __trie_node__(self, ngram, icat):
        """Get the trie's node for this n-gram."""
        try:
            word_info = self.__categories__[icat][VOCAB][ngram[0]]
            for word in ngram[1:]:
                word_info = word_info[NEXT][word]
            return word_info
        except BaseException:
            return EMPTY_WORD_INFO

    def __get_category__(self, name):
        """
        Given the category name, return the category data.

        If category name doesn't exist, creates a new one.
        """
        name = name.lower()
        try:
            return self.__categories_index__[name]
        except KeyError:
            self.__max_fr__.append([])
            self.__max_gv__.append([])
            self.__categories_index__[name] = len(self.__categories__)
            self.__categories__.append([name, {}])  # name, vocabulary
            self.__zero_cv__ = (0,) * len(self.__categories__)
            return self.__categories_index__[name]

    def __get_category_length__(self, icat):
        """
        Return the category length.

        The category length is the total number of words seen during training.
        """
        size = 0
        vocab = self.__categories__[icat][VOCAB]
        for word in vocab:
            size += vocab[word][FR]
        return size

    def __get_most_probable_category__(self):
        """Return the index of the most probable category."""
        sizes = []
        for icat in xrange(len(self.__categories__)):
            sizes.append((icat, self.__get_category_length__(icat)))
        return sorted(sizes, key=lambda v: v[1])[-1][0]

    def __get_vocabularies__(self, icat, vocab, preffix, n_grams, output):
        """Get category list of n-grams with info."""
        senq_ilen = len(preffix)
        get_name = self.get_word

        seq = preffix + [None]
        if len(seq) > n_grams:
            return

        for word in vocab:
            seq[-1] = word
            if (self.__cv__(seq, icat) > 0):
                output[senq_ilen].append(
                    (
                        "_".join([get_name(wi) for wi in seq]),
                        vocab[word][FR],
                        self.__gv__(seq, icat),
                        self.__cv__(seq, icat)
                    )
                )
            self.__get_vocabularies__(
                icat, vocab[word][NEXT], seq, n_grams, output
            )

    def __get_category_vocab__(self, icat):
        """Get category list of n-grams ordered by confidence value."""
        category = self.__categories__[icat]
        vocab = category[VOCAB]
        w_seqs = ([w] for w in vocab)

        vocab_icat = (
            (
                self.get_word(wseq[0]),
                vocab[wseq[0]][FR],
                self.__lv__(wseq, icat),
                self.__gv__(wseq, icat),
                self.__cv__(wseq, icat)
            )
            for wseq in w_seqs if self.__gv__(wseq, icat) > self.__a__
        )
        return sorted(vocab_icat, key=lambda k: -k[-1])

    def __get_next_iwords__(self, sent, icat):
        """Return the list of possible following words' indexes."""
        if not self.get_category_name(icat):
            return []

        vocab = self.__categories__[icat][VOCAB]
        word_index = self.get_word_index
        sent = Pp.clean_and_ready(sent)
        sent = [
            word_index(w)
            for w in sent.strip(".").split(".")[-1].split(" ") if w
        ]

        tips = []
        for word in sent:
            if word is None:
                tips[:] = []
                continue

            tips.append(vocab)

            tips[:] = (
                tip[word][NEXT]
                for tip in tips if word in tip and tip[word][NEXT]
            )

        if len(tips) == 0:
            return []

        next_words = tips[0]
        next_nbr_words = float(sum([next_words[w][FR] for w in next_words]))
        return sorted(
            [
                (
                    word1,
                    next_words[word1][FR],
                    next_words[word1][FR] / next_nbr_words
                )
                for word1 in next_words
            ],
            key=lambda k: -k[1]
        )

    def __prune_cat_trie__(self, vocab, prune=False, min_n=None):
        """Prune the trie of the given category."""
        prun_floor = min_n or self.__prun_floor__
        remove = []
        for word in vocab:
            if prune and vocab[word][FR] <= prun_floor:
                vocab[word][NEXT] = None
                remove.append(word)
            else:
                self.__prune_cat_trie__(vocab[word][NEXT], prune=True)

        for word in remove:
            del vocab[word]

    def __prune_tries__(self):
        """Prune the trie of every category."""
        Print.info("pruning tries...", offset=1)
        for category in self.__categories__:
            self.__prune_cat_trie__(category[VOCAB])
        self.__prun_counter__ = 0

    def __cache_lvs__(self, icat, vocab, preffix):
        """Cache all local values."""
        for word in vocab:
            sequence = preffix + [word]
            vocab[word][LV] = self.__lv__(sequence, icat, cache=False)
            self.__cache_lvs__(icat, vocab[word][NEXT], sequence)

    def __cache_gvs__(self, icat, vocab, preffix):
        """Cache all global values."""
        for word in vocab:
            sequence = preffix + [word]
            vocab[word][GV] = self.__gv__(sequence, icat, cache=False)
            self.__cache_gvs__(icat, vocab[word][NEXT], sequence)

    def __cache_sg__(self, icat, vocab, preffix):
        """Cache all significance weight values."""
        for word in vocab:
            sequence = preffix + [word]
            vocab[word][SG] = self.__sg__(sequence, icat, cache=False)
            self.__cache_sg__(icat, vocab[word][NEXT], sequence)

    def __cache_cvs__(self, icat, vocab, preffix):
        """Cache all confidence values."""
        for word in vocab:
            sequence = preffix + [word]
            vocab[word][CV] = self.__cv__(sequence, icat, False)
            self.__cache_cvs__(icat, vocab[word][NEXT], sequence)

    def __update_max_gvs__(self, icat, vocab, preffix):
        """Update all maximum global values."""
        gv = self.__gv__
        max_gvs = self.__max_gv__[icat]
        sentence_ilength = len(preffix)

        sequence = preffix + [None]
        for word in vocab:
            sequence[-1] = word
            sequence_gv = gv(sequence, icat)
            if sequence_gv > max_gvs[sentence_ilength]:
                max_gvs[sentence_ilength] = sequence_gv
            self.__update_max_gvs__(icat, vocab[word][NEXT], sequence)

    def __update_needed__(self):
        """Return True if an update is needed, false otherwise."""
        return (self.__s__ != self.__s_update__ or
                self.__l__ != self.__l_update__ or
                self.__p__ != self.__p_update__)

    def __save_cat_vocab__(self, icat, path, n_grams):
        """Save the category vocabulary inside ``path``."""
        if n_grams == -1:
            n_grams = 20  # infinite

        category = self.__categories__[icat]
        cat_name = self.get_category_name(icat)
        vocab = category[VOCAB]
        vocabularies_out = [[] for _ in xrange(n_grams)]

        terms = ["words", "bigrams", "trigrams"]

        self.__get_vocabularies__(icat, vocab, [], n_grams, vocabularies_out)

        Print.info("saving '%s' vocab" % cat_name)

        for ilen in xrange(n_grams):
            if vocabularies_out[ilen]:
                term = terms[ilen] if ilen <= 2 else "%d-grams" % (ilen + 1)
                voc_path = os.path.join(
                    path, "ss3_vocab_%s(%s).csv" % (cat_name, term)
                )
                f = open(voc_path, "w+", encoding=ENCODING)
                vocabularies_out[ilen].sort(key=lambda k: -k[-1])
                f.write(u"%s,%s,%s,%s\n" % ("term", "fr", "gv", "norm_gv"))
                for trans in vocabularies_out[ilen]:
                    f.write(u"%s,%d,%f,%f\n" % tuple(trans))
                f.close()
                Print.info("\t[ %s stored in '%s'" % (term, voc_path))

    def summary_op_ngrams(self, cvs):
        """
        Summary operator for n-gram confidence vectors.

        By default it returns the addition of all confidence
        vectors. However, in case you want to use a custom
        summary operator, this function must be replaced
        as shown in the following example:

            >>> def my_summary_op(cvs):
            >>>     return cvs[0]
            >>> ...
            >>> clf = SS3()
            >>> ...
            >>> clf.summary_op_ngrams = my_summary_op

        Note that any function receiving a list of vectors and
        returning a single vector could be used. In the above example
        the summary operator is replaced by the user-defined
        ``my_summary_op`` which ignores all confidence vectors
        returning only the confidence vector of the first n-gram
        (which besides being an illustrative example, makes no real sense).

        :param cvs: a list n-grams confidence vectors
        :type cvs: list (of list of float)
        :returns: a sentence confidence vector
        :rtype: list (of float)
        """
        return reduce(vsum, cvs)

    def summary_op_sentences(self, cvs):
        """
        Summary operator for sentence confidence vectors.

        By default it returns the addition of all confidence
        vectors. However, in case you want to use a custom
        summary operator, this function must be replaced
        as shown in the following example:

            >>> def dummy_summary_op(cvs):
            >>>     return cvs[0]
            >>> ...
            >>> clf = SS3()
            >>> ...
            >>> clf.summary_op_sentences = dummy_summary_op

        Note that any function receiving a list of vectors and
        returning a single vector could be used. In the above example
        the summary operator is replaced by the user-defined
        ``dummy_summary_op`` which ignores all confidence vectors
        returning only the confidence vector of the first sentence
        (which besides being an illustrative example, makes no real sense).

        :param cvs: a list sentence confidence vectors
        :type cvs: list (of list of float)
        :returns: a paragraph confidence vector
        :rtype: list (of float)
        """
        return reduce(vsum, cvs)

    def summary_op_paragraphs(self, cvs):
        """
        Summary operator for paragraph confidence vectors.

        By default it returns the addition of all confidence
        vectors. However, in case you want to use a custom
        summary operator, this function must be replaced
        as shown in the following example:

            >>> def dummy_summary_op(cvs):
            >>>     return cvs[0]
            >>> ...
            >>> clf = SS3()
            >>> ...
            >>> clf.summary_op_paragraphs = dummy_summary_op

        Note that any function receiving a list of vectors and
        returning a single vector could be used. In the above example
        the summary operator is replaced by the user-defined
        ``dummy_summary_op`` which ignores all confidence vectors
        returning only the confidence vector of the first paragraph
        (which besides being an illustrative example, makes no real sense).

        :param cvs: a list paragraph confidence vectors
        :type cvs: list (of list of float)
        :returns: the document confidence vector
        :rtype: list (of float)
        """
        return reduce(vsum, cvs)

    def get_name(self):
        """
        Return the model's name.

        :returns: the model's name.
        :rtype: str
        """
        return self.__name__

    def set_hyperparameters(self, s=None, l=None, p=None, a=None):
        """
        Set hyperparameter values.

        :param s: the "smoothness" (sigma) hyperparameter
        :type s: float
        :param l: the "significance" (lambda) hyperparameter
        :type l: float
        :param p: the "sanction" (rho) hyperparameter
        :type p: float
        :param a: the alpha hyperparameter (i.e. all terms with a
                  confidence value (cv) less than alpha will be ignored during
                  classification)
        :type a: float
        """
        if s is not None:
            self.set_s(s)
        if l is not None:
            self.set_l(l)
        if p is not None:
            self.set_p(p)
        if a is not None:
            self.set_a(a)

    def get_hyperparameters(self):
        """
        Get hyperparameter values.

        :returns: a tuple with hyperparameters current values (s, l, p, a)
        :rtype: tuple
        """
        return self.__s__, self.__l__, self.__p__, self.__a__

    def set_model_path(self, path):
        """
        Overwrite the default path from which the model will be loaded (or saved to).

        Note: be aware that the PySS3 Command Line tool looks for
        a local folder called ``ss3_models`` to load models.
        Therefore, the ``ss3_models`` folder will be always automatically
        append to the given ``path`` (e.g. if ``path="my/path/"``, it will
        be converted into ``my/path/ss3_models``).

        :param path: the path
        :type path: str
        """
        self.__models_folder__ = os.path.join(path, STR_MODEL_FOLDER)

    def set_block_delimiters(self, parag=None, sent=None, word=None):
        r"""Overwrite the default delimiters used to split input documents into blocks.

        delimiters are any regular expression from simple ones (e.g. ``" "``) to
        more complex ones (e.g. ``r"[^\s\w\d]"``).
        Note: remember that there are certain reserved characters for regular expression,
              for example, the dot (.), in which case use the backslash to indicate you're
              referring the character itself and not its interpretation (e.g. ``\.``)

        e.g.

        >>> ss3.set_block_delimiters(word="\s")
        >>> ss3.set_block_delimiters(word="\s", parag="\n\n")
        >>> ss3.set_block_delimiters(parag="\n---\n")
        >>> ss3.set_block_delimiters(sent="\.")
        >>> ss3.set_block_delimiters(word="\|")
        >>> ss3.set_block_delimiters(word=" ")

        :param parag: the paragraph new delimiter
        :type parag: str
        :param sent: the sentence new delimiter
        :type sent: str
        :param word: the word new delimiter
        :type word: str
        """
        if parag:
            self.set_delimiter_paragraph(parag)
        if sent:
            self.set_delimiter_sentence(sent)
        if word:
            self.set_delimiter_word(word)

    def set_delimiter_paragraph(self, regex):
        r"""
        Set the delimiter used to split documents into paragraphs.

        Remember that there are certain reserved characters for regular expression,
        for example, the dot (.), in which case use the backslash to indicate you're
        referring the character itself and not its interpretation (e.g. ``\.``)

        :param regex: the regular expression of the new delimiter
        :type regex: str
        """
        self.__parag_delimiter__ = regex

    def set_delimiter_sentence(self, regex):
        r"""
        Set the delimiter used to split documents into sentences.

        Remember that there are certain reserved characters for regular expression,
        for example, the dot (.), in which case use the backslash to indicate you're
        referring the character itself and not its interpretation (e.g. ``\.``)

        :param regex: the regular expression of the new delimiter
        :type regex: str
        """
        self.__sent_delimiter__ = regex

    def set_delimiter_word(self, regex):
        r"""
        Set the delimiter used to split documents into words.

        Remember that there are certain reserved characters for regular expression,
        for example, the dot (.), in which case use the backslash to indicate you're
        referring the character itself and not its interpretation (e.g. ``\.``)

        :param regex: the regular expression of the new delimiter
        :type regex: str
        """
        self.__word_delimiter__ = regex

    def set_s(self, value):
        """
        Set the "smoothness" (sigma) hyperparameter value.

        :param value: the hyperparameter value
        :type value: float
        """
        self.__s__ = float(value)

    def get_s(self):
        """
        Get the "smoothness" (sigma) hyperparameter value.

        :returns: the hyperparameter value
        :rtype: float
        """
        return self.__s__

    def set_l(self, value):
        """
        Set the "significance" (lambda) hyperparameter value.

        :param value: the hyperparameter value
        :type value: float
        """
        self.__l__ = float(value)

    def get_l(self):
        """
        Get the "significance" (lambda) hyperparameter value.

        :returns: the hyperparameter value
        :rtype: float
        """
        return self.__l__

    def set_p(self, value):
        """
        Set the "sanction" (rho) hyperparameter value.

        :param value: the hyperparameter value
        :type value: float
        """
        self.__p__ = float(value)

    def get_p(self):
        """
        Get the "sanction" (rho) hyperparameter value.

        :returns: the hyperparameter value
        :rtype: float
        """
        return self.__p__

    def set_a(self, value):
        """
        Set the alpha hyperparameter value.

        All terms with a confidence value (cv) less than alpha
        will be ignored during classification.

        :param value: the hyperparameter value
        :type value: float
        """
        self.__a__ = float(value)

    def get_a(self):
        """
        Get the alpha hyperparameter value.

        :returns: the hyperparameter value
        :rtype: float
        """
        return self.__a__

    def get_categories(self):
        """
        Get the list of category names.

        :returns: the list of category names
        :rtype: list (of str)
        """
        return [
            self.get_category_name(ci)
            for ci in range(len(self.__categories__))
        ]

    def get_most_probable_category(self):
        """
        Get the name of the most probable category.

        :returns: the name of the most probable category
        :rtype: str
        """
        return self.get_category_name(self.__get_most_probable_category__())

    def get_category_index(self, name):
        """
        Given its name, return the category index.

        :param name: The category name
        :type name: str
        :returns: the category index (or ``IDX_UNKNOWN_CATEGORY``
                  if the category doesn't exist).
        :rtype: int
        """
        try:
            if type(name) is str:
                name = name.lower()
            return self.__categories_index__[name]
        except KeyError:
            return IDX_UNKNOWN_CATEGORY

    def get_category_name(self, index):
        """
        Given its index, return the category name.

        :param index: The category index
        :type index: int
        :returns: the category name (or ``STR_UNKNOWN_CATEGORY``
                  if the category doesn't exist).
        :rtype: str
        """
        try:
            if type(index) == list:
                index = index[0]
            return self.__categories__[index][NAME]
        except IndexError:
            return STR_UNKNOWN_CATEGORY

    def get_word_index(self, word):
        """
        Given a word, return its index.

        :param name: a word
        :type name: str
        :returns: the word index (or ``IDX_UNKNOWN_WORD`` if the word doesn't exist).
        :rtype: int
        """
        try:
            return self.__word_to_index__[word]
        except KeyError:
            return IDX_UNKNOWN_WORD

    def get_word(self, index):
        """
        Given the index, return the word.

        :param index: the word index
        :type index: int
        :returns: the word (or ``STR_UNKNOWN_WORD`` if the word doesn't exist).
        :rtype: int
        :rtype: str
        """
        return (
            self.__index_to_word__[index]
            if index in self.__index_to_word__ else STR_UNKNOWN_WORD
        )

    def get_next_words(self, sent, cat, n=None):
        """
        Given a sentence, return the list of ``n`` (possible) following words.

        :param sent: a sentence (e.g. "an artificial")
        :type sent: str
        :param cat: the category name
        :type cat: str
        :param n: the maximum number of possible answers
        :type n: int
        :returns: a list of tuples (word, frequency, probability)
        :rtype: list (of tuple)
        :raises: InvalidCategoryError
        """
        icat = self.get_category_index(cat)

        if icat == IDX_UNKNOWN_CATEGORY:
            raise InvalidCategoryError

        guessedwords = [
            (self.get_word(iword), fr, P)
            for iword, fr, P in self.__get_next_iwords__(sent, icat) if fr
        ]
        if n is not None and guessedwords:
            return guessedwords[:n]
        return guessedwords

    def get_stopwords(self, sg_threshold=.01):
        """
        Get the list of (recognized) stopwords.

        :param sg_threshold: significance (sg) value used as a threshold to
                             consider words as stopwords (i.e. words with
                             sg < ``sg_threshold`` for all categories will
                             be considered as "stopwords")
        :type sg_threshold: float
        :returns: a list of stopwords
        :rtype: list (of str)
        """
        if not self.__categories__:
            return

        iwords = self.__index_to_word__
        sg_threshold = float(sg_threshold or .01)
        categories = self.__categories__
        cats_len = len(categories)
        sg = self.__sg__
        stopwords = []
        vocab = categories[0][VOCAB]

        for word0 in iwords:
            word_sg = [
                sg([word0], c_i)
                for c_i in xrange(cats_len)
            ]
            word_cats_len = len([v for v in word_sg if v < sg_threshold])
            if word_cats_len == cats_len:
                stopwords.append(word0)

        stopwords = [
            iwords[w0]
            for w0, v
            in sorted(
                [
                    (w0, vocab[w0][FR] if w0 in vocab else 0)
                    for w0 in stopwords
                ],
                key=lambda k: -k[1]
            )
        ]

        return stopwords

    def save_model(self, path=None):
        """
        Save the model to disk.

        if a ``path`` is not present, the default will be used ("./"),
        However, if a ``path`` is given, it will not only used to save
        the model but also will overwrite the default path calling the
        ``SS3``'s ``set_model_path(path)`` method (see ``set_model_path``
        method documentation for more detail).

        :param path: the path to save the model to
        :type path: str

        :raises: OSError
        """
        if path:
            self.set_model_path(path)

        stime = time()
        Print.info(
            "saving model (%s/%s.%s)..."
            %
            (self.__models_folder__, self.__name__, STR_MODEL_EXT),
            False
        )
        json_file_format = {
            "__a__": self.__a__,
            "__l__": self.__l__,
            "__p__": self.__p__,
            "__s__": self.__s__,
            "__max_fr__": self.__max_fr__,
            "__max_gv__": self.__max_gv__,
            "__categories__": self.__categories__,
            "__categories_index__": self.__categories_index__,
            "__index_to_word__": self.__index_to_word__,
            "__word_to_index__": self.__word_to_index__,
            "__cv_mode__": self.__cv_mode__,
            "__sn_mode__": self.__sn_mode__
        }

        try:
            os.makedirs(self.__models_folder__)
        except OSError as ose:
            if ose.errno == errno.EEXIST and os.path.isdir(self.__models_folder__):
                pass
            else:
                raise

        json_file = open(
            "%s/%s.%s" % (
                self.__models_folder__,
                self.__name__,
                STR_MODEL_EXT
            ), "w", encoding=ENCODING
        )

        try:  # python 3
            json_file.write(json.dumps(json_file_format))
        except TypeError:  # python 2
            json_file.write(json.dumps(json_file_format).decode(ENCODING))

        json_file.close()
        Print.info("(%.1fs)" % (time() - stime))

    def load_model(self, path=None):
        """
        Load model from disk.

        if a ``path`` is not present, the default will be used ("./"),
        However, if a ``path`` is given, it will not only used to load
        the model but also will overwrite the default path calling the
        ``SS3``'s ``set_model_path(path)`` method (see ``set_model_path``
        method documentation for more detail).

        :param path: the path to load the model from
        :type path: str

        :raises: OSError
        """
        if path:
            self.set_model_path(path)

        stime = time()
        Print.info("loading '%s' model from disk..." % self.__name__)

        json_file = open(
            "%s/%s.%s" % (
                self.__models_folder__,
                self.__name__,
                STR_MODEL_EXT
            ), "r", encoding=ENCODING
        )
        json_file_format = json.loads(json_file.read(), object_hook=key_as_int)
        json_file.close()

        self.__max_fr__ = json_file_format["__max_fr__"]
        self.__max_gv__ = json_file_format["__max_gv__"]
        self.__l__ = json_file_format["__l__"]
        self.__p__ = json_file_format["__p__"]
        self.__s__ = json_file_format["__s__"]
        self.__a__ = json_file_format["__a__"]
        self.__categories__ = json_file_format["__categories__"]
        self.__categories_index__ = json_file_format["__categories_index__"]
        self.__index_to_word__ = json_file_format["__index_to_word__"]
        self.__word_to_index__ = json_file_format["__word_to_index__"]
        self.__cv_mode__ = json_file_format["__cv_mode__"]
        self.__sn_mode__ = json_file_format["__sn_mode__"]

        self.__zero_cv__ = (0,) * len(self.__categories__)
        self.__s_update__ = self.__s__
        self.__l_update__ = self.__l__
        self.__p_update__ = self.__p__

        Print.info("(%.1fs)" % (time() - stime))

    def save_cat_vocab(self, cat, path="./", n_grams=-1):
        """
        Save category vocabulary to disk.

        :param cat: the category name
        :type cat: str
        :param path: the path in which to store the vocabulary
        :type path: str
        :param n_grams: indicates the n-grams to be stored (e.g. only 1-grams,
                        2-grams, 3-grams, etc.). Default -1 stores all
                        learned n-grams (1-grams, 2-grams, 3-grams, etc.)
        :type n_grams: int
        :raises: InvalidCategoryError
        """
        if self.get_category_index(cat) == IDX_UNKNOWN_CATEGORY:
            raise InvalidCategoryError

        self.__save_cat_vocab__(self.get_category_index(cat), path, n_grams)

    def save_vocab(self, path="./", n_grams=-1):
        """
        Save learned vocabularies to disk.

        :param path: the path in which to store the vocabularies
        :type path: str
        :param n_grams: indicates the n-grams to be stored (e.g. only 1-grams,
                        2-grams, 3-grams, etc.). Default -1 stores all
                        learned n-grams (1-grams, 2-grams, 3-grams, etc.)
        :type n_grams: int
        """
        for icat in xrange(len(self.__categories__)):
            self.__save_cat_vocab__(icat, path, n_grams)

    def update_values(self, force=False):
        """
        Update model values (cv, gv, lv, etc.).

        :param force: force update (even if hyperparameters haven't changed)
        :type force: bool
        """
        update = 0
        if force or self.__s_update__ != self.__s__:
            update = 3
        elif self.__l_update__ != self.__l__:
            update = 2
        elif self.__p_update__ != self.__p__:
            update = 1

        if update == 0:
            Print.info("nothing to update...", offset=1)
            return

        category_len = len(self.__categories__)
        categories = xrange(category_len)
        category_names = [self.get_category_name(ic) for ic in categories]
        stime = time()
        Print.info("about to start updating values...", offset=1)
        if update == 3:  # only if s has changed
            Print.info("caching lv values", offset=1)
            for icat in categories:
                Print.info(
                    "lv values for %d (%s)" % (icat, category_names[icat]),
                    offset=4
                )
                self.__cache_lvs__(icat, self.__categories__[icat][VOCAB], [])

        if update >= 2:  # only if s or l have changed
            Print.info("caching sg values", offset=1)
            for icat in categories:
                Print.info(
                    "sg values for %d (%s)" % (icat, category_names[icat]),
                    offset=4
                )
                self.__cache_sg__(icat, self.__categories__[icat][VOCAB], [])

        Print.info("caching gv values")
        for icat in categories:
            Print.info(
                "gv values for %d (%s)" % (icat, category_names[icat]),
                offset=4
            )
            self.__cache_gvs__(icat, self.__categories__[icat][VOCAB], [])

        if self.__cv_mode__ != STR_GV:
            Print.info("updating max gv values", offset=1)
            for icat in categories:
                Print.info(
                    "max gv values for %d (%s)" % (icat, category_names[icat]),
                    offset=4
                )
                self.__max_gv__[icat] = list(
                    map(lambda _: 0, self.__max_gv__[icat])
                )
                self.__update_max_gvs__(
                    icat, self.__categories__[icat][VOCAB], []
                )

            Print.info("max gv values have been updated", offset=1)

            Print.info("caching confidence values (cvs)", offset=1)
            for icat in categories:
                Print.info(
                    "cvs for %d (%s)" % (icat, category_names[icat]),
                    offset=4
                )
                self.__cache_cvs__(icat, self.__categories__[icat][VOCAB], [])
        Print.info("finished --time: %.1fs" % (time() - stime), offset=1)

        self.__s_update__ = self.__s__
        self.__l_update__ = self.__l__
        self.__p_update__ = self.__p__

    def print_model_info(self):
        """Print information regarding the model."""
        print()
        print(" %s: %s\n" % (
            Print.style.green(Print.style.ubold("NAME")),
            Print.style.warning(self.get_name())
        ))

    def print_hyperparameters_info(self):
        """Print information about hyperparameters."""
        print()
        print(
            " %s:\n" % Print.style.green(Print.style.ubold("HYPERPARAMETERS"))
        )
        print("\tSmoothness(s):", Print.style.warning(self.__s__))
        print("\tSignificance(l):", Print.style.warning(self.__l__))
        print("\tSanction(p):", Print.style.warning(self.__p__))
        print("")
        print("\tAlpha(a):", Print.style.warning(self.__a__))

    def print_categories_info(self):
        """Print information about learned categories."""
        if not self.__categories__:
            print(
                "\n %s: None\n"
                % Print.style.green(Print.style.ubold("CATEGORIES"))
            )
            return

        cat_len = max([
            len(self.get_category_name(ic))
            for ic in xrange(len(self.__categories__))
        ])
        cat_len = max(cat_len, 8)
        row_template = Print.style.warning("\t{:^%d} " % cat_len)
        row_template += "| {:^5} | {:^10} | {:^11} | {:^13} | {:^6} |"
        print()
        print("\n %s:\n" % Print.style.green(Print.style.ubold("CATEGORIES")))
        print(
            row_template
            .format(
                "Category", "Index", "Length",
                "Vocab. Size", "Word Max. Fr.", "N-gram"
            )
        )
        print(
            (
                "\t{:-<%d}-|-{:-<5}-|-{:-<10}-|-{:-<11}-|-{:-<13}-|-{:-<6}-|"
                % cat_len
            )
            .format('', '', '', '', '', '')
        )

        mpci = self.__get_most_probable_category__()
        mpc_size = 0
        mpc_total = 0
        for icat, category in enumerate(self.__categories__):
            icat_size = self.__get_category_length__(icat)
            print(
                row_template
                .format(
                    category[NAME],
                    icat, icat_size,
                    len(category[VOCAB]),
                    self.__max_fr__[icat][0],
                    len(self.__max_fr__[icat])
                )
            )

            mpc_total += icat_size
            if icat == mpci:
                mpc_size = icat_size

        print(
            "\n\t%s: %s %s"
            %
            (
                Print.style.ubold("Most Probable Category"),
                Print.style.warning(self.get_category_name(mpci)),
                Print.style.blue("(%.2f%%)" % (100.0 * mpc_size / mpc_total))
            )
        )
        print()

    def print_ngram_info(self, ngram):
        """
        Print debugging information about a given n-gram.

        Namely, print the n-gram frequency (fr), local value (lv),
        global value (gv), confidence value (cv), sanction (sn) weight,
        significance (sg) weight.

        :param ngram: the n-gram (e.g. "machine", "machine learning", etc.)
        :type ngram: str
        """
        if not self.__categories__:
            return

        word_index = self.get_word_index
        n_gram_str = ngram
        ngram = [word_index(w)
                 for w in re.split(self.__word_delimiter__, ngram)
                 if w]

        print()
        print(
            " %s: %s (%s)" % (
                Print.style.green(
                    "%d-GRAM" % len(ngram) if len(ngram) > 1 else "WORD"
                ),
                Print.style.warning(n_gram_str),
                "is unknown"
                if IDX_UNKNOWN_WORD in ngram
                else "index: " + str(ngram if len(ngram) > 1 else ngram[0])
            )
        )

        if IDX_UNKNOWN_WORD in ngram:
            print()
            return

        cat_len = max([
            len(self.get_category_name(ic))
            for ic in xrange(len(self.__categories__))
        ])
        cat_len = max(cat_len, 8)
        header_template = Print.style.bold(
            " {:<%d} |    fr    |  lv   |  sg   |  sn   |  gv   |  cv   |"
            % cat_len
        )
        print()
        print(header_template.format("Category"))
        header_template = (
            " {:-<%d}-|----------|-------|-------|-------|-------|-------|"
            % cat_len
        )
        print(header_template.format(''))
        row_template = (
            " %s | {:^8} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |"
            % (Print.style.warning("{:<%d}" % cat_len))
        )
        for icat in xrange(len(self.__categories__)):
            n_gram_tip = self.__trie_node__(ngram, icat)
            if n_gram_tip:
                print(
                    row_template
                    .format(
                        self.get_category_name(icat)[:16],
                        n_gram_tip[FR],
                        self.__lv__(ngram, icat),
                        self.__sg__(ngram, icat),
                        self.__sn__(ngram, icat),
                        self.__gv__(ngram, icat),
                        self.__cv__(ngram, icat),
                    )
                )
        print()

    def plot_value_distribution(self, cat):
        """
        Plot the category's global and local value distribution.

        :param cat: the category name
        :type cat: str
        :raises: InvalidCategoryError
        """
        if self.get_category_index(cat) == IDX_UNKNOWN_CATEGORY:
            raise InvalidCategoryError

        import matplotlib.pyplot as plt

        icat = self.get_category_index(cat)
        vocab_metrics = self.__get_category_vocab__(icat)

        x = []
        y_lv = []
        y_gv = []
        vocab_metrics_len = len(vocab_metrics)

        for i in xrange(vocab_metrics_len):
            metric = vocab_metrics[i]
            x.append(i + 1)
            y_lv.append(metric[2])
            y_gv.append(metric[3])

        plt.figure(figsize=(20, 10))
        plt.title(
            "Word Value Distribution (%s)" % self.get_category_name(icat)
        )

        plt.xlabel("Word Rank")
        plt.ylabel("Value")
        plt.xlim(right=max(x))

        plt.plot(
            x, y_lv, "-", label="local value ($lv$)",
            linewidth=2, color="#7f7d7e"
        )
        plt.plot(
            x, y_gv, "g-", label="global value ($gv$)",
            linewidth=4, color="#2ca02c")
        plt.legend()

        plt.show()

    def extract_insight(
        self, doc, cat='auto', level='word', window_size=3, min_cv=0.01, sort=True
    ):
        """
        Get the list of text blocks involved in the classification decision.

        Given a document, return the pieces of text that were involved in the
        classification decision, along with the confidence values associated
        with them. If a category is given, perform the process as if the
        given category were the one assigned by the classifier.

        :param doc: the content of the document
        :type doc: str
        :param cat: the category in relation to which text blocks are obtained.
                    If not present, it will automatically use the category assigned
                    by SS3 after classification.
                    Options are 'auto' or a given category name. (default: 'auto')
        :type cat: str
        :param level: the level at which text blocks are going to be extracted.
                      Options are 'word', 'sentence' or 'paragraph'. (default: 'word')
        :type level: str
        :param window_size: the number of words, before and after each identified word,
                            to be also included along with the identified word. For instance,
                            ``window_size=0`` means return only individual words,
                            ``window_size=1`` means  also include the word that was
                            before and the one that was after them. If multiple selected
                            words are close enough for their word windows to be overlapping,
                            then those word windows will be merged into a longer and single one.
                            This argument is ignored when ``level`` is not equal to 'word'.
                            (default: 3)
        :type window_size: int
        :param min_cv: the minimum confidence value each text block must have to be
                       included in the output. (default 0.01)
        :type min_cv: float
        :param sort: whether to return the text blocks ordered by their confidence value
                     or not. If ``sort=False`` then blocks will be returned
                     following the order they had in the input document. (default: True)
        :type sort: bool
        :returns: a list of pairs (text, confidence value) containing the text (blocks) involved,
                  and to what degree (*), in the classification decision.
                  (*) given by the confidence value
        :rtype: list
        :raises: InvalidCategoryError, ValueError
        """
        r = self.classify(doc, json=True)

        if cat == 'auto':
            c_i = r["cvns"][0][0]
        else:
            c_i = self.get_category_index(cat)
            if c_i == IDX_UNKNOWN_CATEGORY:
                Print.error(
                    "The excepted values for the `cat` argument are 'auto' "
                    "or a valid category name, found '%s' instead" % str(cat),
                    raises=InvalidCategoryError
                )

        if level == 'paragraph':
            insights = [
                (
                    "".join([word["lexeme"]
                             for s in p["sents"]
                             for word in s["words"]]),
                    p["cv"][c_i]
                )
                for p in r["pars"]
                if p["cv"][c_i] > min_cv
            ]
        elif level == 'sentence':
            insights = [
                (
                    "".join([word["lexeme"]
                             for word in s["words"]]),
                    s["cv"][c_i]
                )
                for p in r["pars"] for s in p["sents"]
                if s["cv"][c_i] > min_cv
            ]
        elif level == 'word':
            ww_size = window_size
            insights = []
            for p in r["pars"]:
                words = [w for s in p["sents"] for w in s["words"]]
                w_i = 0
                while w_i < len(words):
                    w = words[w_i]
                    if w["cv"][c_i] > min_cv:
                        ww = []
                        ww_cv = 0
                        ww_left = min(w_i, ww_size) + 1
                        w_i -= ww_left - 1
                        while ww_left > 0 and w_i < len(words):

                            ww.append(words[w_i]["lexeme"])
                            ww_cv += words[w_i]["cv"][c_i]

                            if words[w_i]["cv"][c_i] > min_cv:
                                ww_left += min(ww_size, (len(words) - 1) - w_i)

                            if re.search(r"[\w\d]+", words[w_i]["lexeme"]):
                                ww_left -= 1

                            w_i += 1

                        insights.append(("".join(ww), ww_cv))
                    else:
                        w_i += 1
        else:
            raise ValueError(
                "expected values for the `level` argument are "
                "'word', 'sentence', or 'paragraph', found '%s' instead."
                % str(level)
            )

        if sort:
            insights.sort(key=lambda b_cv: -b_cv[1])
        return insights

    def learn(self, doc, cat, n_grams=1, prep=True, update=True):
        """
        Learn a new document for a given category.

        :param doc: the content of the document
        :type doc: str
        :param cat: the category name
        :type cat: str
        :param n_grams: indicates the maximum ``n``-grams to be learned
                        (e.g. a value of ``1`` means only 1-grams (words),
                        ``2`` means 1-grams and 2-grams,
                        ``3``, 1-grams, 2-grams and 3-grams, and so on.
        :type n_grams: int
        :param prep: enables input preprocessing (default: True)
        :type prep: bool
        :param update: enables model auto-update after learning (default: True)
        :type update: bool
        """
        try:
            doc = doc.decode(ENCODING)
        except AttributeError:
            pass
        try:
            icat = self.__get_category__(cat)
        except AttributeError:
            icat = self.__get_category__(cat)

        cat = self.__categories__[icat]
        word_to_index = self.__word_to_index__

        if prep:
            Print.info("preprocessing document...", offset=1)
            stime = time()
            doc = Pp.clean_and_ready(doc)
            Print.info("finished --time: %.1fs" % (time() - stime), offset=1)
        doc = doc.replace("\n", "").split(" ")

        text_len = len(doc)
        Print.info(
            "about to learn new document (%d terms)" % text_len, offset=1
        )

        vocab = cat[VOCAB]  # getting cat vocab

        index_to_word = self.__index_to_word__
        max_frs = self.__max_fr__[icat]
        max_gvs = self.__max_gv__[icat]

        stime = time()
        Print.info("learning...", offset=1)
        tips = []
        for word in doc:
            if word != ".":
                self.__prun_counter__ += 1
                # if word doesn't exist yet, then...
                try:
                    word = word_to_index[word]
                except KeyError:
                    new_index = len(word_to_index)
                    word_to_index[word] = new_index
                    index_to_word[new_index] = word
                    word = new_index

                tips.append(vocab)

                if len(tips) > n_grams:
                    del tips[0]

                tips_length = len(tips)

                for i in xrange(tips_length):
                    tips_i = tips[i]

                    try:
                        max_frs[i]
                    except IndexError:
                        max_frs.append(1)
                        max_gvs.append(0)

                    try:
                        word_info = tips_i[word]
                        word_info[FR] += 1

                        if word_info[FR] > max_frs[(tips_length - 1) - i]:
                            max_frs[(tips_length - 1) - i] = word_info[FR]
                    except KeyError:
                        tips_i[word] = [
                            {},  # NEXT/VOCAB
                            1,   # FR
                            0,   # CV
                            0,   # SG
                            0,   # GV
                            0    # LV
                        ]
                        word_info = tips_i[word]

                    # print i, index_to_word[ word ], tips_i[word][FR]
                    tips[i] = word_info[NEXT]
            else:
                tips[:] = []
                if self.__prun_counter__ >= self.__prun_trigger__:
                    # trie data-structures pruning
                    self.__prune_tries__()

        Print.info("finished --time: %.1fs" % (time() - stime), offset=1)
        # updating values
        if update:
            self.update_values(force=True)

    def classify(self, doc, prep=True, sort=True, json=False):
        """
        Classify a given document.

        :param doc: the content of the document
        :type doc: str
        :param prep: enables input preprocessing (default: True)
        :type prep: bool
        :param sort: sort the classification result (from best to worst)
        :type sort: bool
        :param json: return a debugging version of the result in JSON format.
        :type json: bool
        :returns: the document confidence vector if ``sort`` is False.
                  If ``sort`` is True, a list of pairs
                  (category index, confidence value) ordered by confidence value.
        :rtype: list
        """
        if not self.__categories__ or not doc:
            return []

        if self.__update_needed__():
            self.update_values()

        try:
            doc = doc.decode(ENCODING)
        except BaseException:
            pass

        if not json:
            paragraphs_cvs = [
                self.__classify_paragraph__(parag, prep=prep)
                for parag in re.split(self.__parag_delimiter__, doc)
                if parag
            ]
            if paragraphs_cvs:
                cv = self.summary_op_paragraphs(paragraphs_cvs)
            else:
                cv = self.__zero_cv__
            if sort:
                return sorted(
                    [
                        (i, cv[i])
                        for i in xrange(len(cv))
                    ],
                    key=lambda e: -e[1]
                )
            return cv
        else:
            info = [
                self.__classify_paragraph__(parag, prep=prep, json=True)
                for parag in re_split_keep(self.__parag_delimiter__, doc)
                if parag
            ]

            nbr_cats = len(self.__categories__)
            cv = self.summary_op_paragraphs([v["cv"] for v in info])
            max_v = max(cv)

            if max_v > 1:
                norm_cv = map(lambda x: x / max_v, cv)
            else:
                norm_cv = cv

            norm_cv_sorted = sorted(
                [(i, nv, cv[i]) for i, nv in enumerate(norm_cv)],
                key=lambda e: -e[1]
            )

            return {
                "pars": info,
                "cv": cv,
                "wmv": reduce(vmax, [v["wmv"] for v in info]),
                "cvns": norm_cv_sorted,
                "ci": [self.get_category_name(ic) for ic in xrange(nbr_cats)]
            }

    def classify_label(self, doc, def_cat=STR_MOST_PROBABLE, labels=True, prep=True):
        """
        Classify a given document returning the category label.

        :param doc: the content of the document
        :type doc: str
        :param def_cat: default category to be assigned when SS3 is not
                        able to classify a document. Options are
                        "most-probable", "unknown" or a given category name.
                        (default: "most-probable")
        :type def_cat: str
        :param labels: whether to return the category label or just the
                       category index (default: True)
        :type labels: bool
        :param prep: enables input preprocessing (default: True)
        :type prep: bool
        :returns: the category label or the category index.
        :rtype: str or int
        :raises: InvalidCategoryError
        """
        r = self.classify(doc, sort=True, prep=prep)

        if not r[0][1]:
            if not def_cat or def_cat == STR_UNKNOWN:
                cat = STR_UNKNOWN_CATEGORY
            elif def_cat == STR_MOST_PROBABLE:
                cat = self.get_most_probable_category()
            else:
                if self.get_category_index(def_cat) == IDX_UNKNOWN_CATEGORY:
                    raise InvalidCategoryError
                cat = def_cat
        else:
            cat = self.get_category_name(r[0][0])

        return cat if labels else self.get_category_index(cat)

    def classify_multilabel(self, doc, def_cat=STR_MOST_PROBABLE, labels=True, prep=True):
        """
        Classify a given document returning multiple category labels.

        This method could be used to perform multi-label classification. Internally, it
        uses k-mean clustering on the confidence vector to select the proper group of
        labels.

        :param doc: the content of the document
        :type doc: str
        :param def_cat: default category to be assigned when SS3 is not
                        able to classify a document. Options are
                        "most-probable", "unknown" or a given category name.
                        (default: "most-probable")
        :type def_cat: str
        :param labels: whether to return the category labels or just the
                       category indexes (default: True)
        :type labels: bool
        :param prep: enables input preprocessing (default: True)
        :type prep: bool
        :returns: the list of category labels (or indexes).
        :rtype: list (of str or int)
        :raises: InvalidCategoryError
        """
        r = self.classify(doc, sort=True, prep=prep)

        if not r[0][1]:
            if not def_cat or def_cat == STR_UNKNOWN:
                cat = STR_UNKNOWN_CATEGORY
            elif def_cat == STR_MOST_PROBABLE:
                cat = self.get_most_probable_category()
            else:
                if self.get_category_index(def_cat) == IDX_UNKNOWN_CATEGORY:
                    raise InvalidCategoryError
                cat = def_cat
            return [cat] if labels else [self.get_category_index(cat)]
        else:
            if labels:
                return [
                    self.get_category_name(cat_i)
                    for cat_i, _ in r[:kmean_multilabel_size(r)]
                ]
            else:
                return [cat_i for cat_i, _ in r[:kmean_multilabel_size(r)]]

    def fit(self, x_train, y_train, n_grams=1, prep=True, leave_pbar=True):
        """
        Train the model given a list of documents and category labels.

        :param x_train: the list of documents
        :type x_train: list (of str)
        :param y_train: the list of document labels
        :type y_train: list (of str)
        :param n_grams:  indicates the maximum ``n``-grams to be learned
                        (e.g. a value of ``1`` means only 1-grams (words),
                        ``2`` means 1-grams and 2-grams,
                        ``3``, 1-grams, 2-grams and 3-grams, and so on.
        :type n_grams: int
        :param prep: enables input preprocessing (default: True)
        :type prep: bool
        :param leave_pbar: controls whether to leave the progress bar or
                           remove it after finishing.
        :type leave_pbar: bool
        """
        cats = sorted(list(set(y_train)))
        stime = time()

        x_train = [
            "".join([
                x_train[i] if x_train[i][-1] == '\n' else x_train[i] + '\n'
                for i in xrange(len(x_train))
                if y_train[i] == cat
            ])
            for cat in cats
        ]
        y_train = list(cats)

        Print.info("about to start training", offset=1)
        for i in tqdm(range(len(x_train)), desc=" Training",
                      leave=leave_pbar, disable=Print.is_quiet()):
            self.learn(
                x_train[i], y_train[i],
                n_grams=n_grams, prep=prep, update=False
            )
        self.__prune_tries__()
        Print.info("finished --time: %.1fs" % (time() - stime), offset=1)
        self.update_values(force=True)

    def predict_proba(self, x_test, prep=True, leave_pbar=True):
        """
        Classify a list of documents returning a list of confidence vectors.

        :param x_test: the list of documents to be classified
        :type x_test: list (of str)
        :param prep: enables input preprocessing (default: True)
        :type prep: bool
        :param leave_pbar: controls whether to leave the progress bar after
                           finishing or remove it.
        :type leave_pbar: bool
        :returns: the list of confidence vectors
        :rtype: list (of list of float)
        :raises: EmptyModelError
        """
        if not self.__categories__:
            raise EmptyModelError

        classify = self.classify
        return [
            classify(x, sort=False)
            for x in tqdm(x_test, desc=" Classification", disable=Print.is_quiet())
        ]

    def predict(
        self, x_test, def_cat=STR_MOST_PROBABLE,
        labels=True, multilabel=False, prep=True, leave_pbar=True
    ):
        """
        Classify a list of documents.

        :param x_test: the list of documents to be classified
        :type x_test: list (of str)
        :param def_cat: default category to be assigned when SS3 is not
                        able to classify a document. Options are
                        "most-probable", "unknown" or a given category name.
        :type def_cat: str
        :param labels: whether to return the list of category names or just
                       category indexes
        :type labels: bool
        :param multilabel: whether to perform multi-label classification or not.
                           if enabled, for each document returns a ``list`` of labels
                           instead of a single label (``str``).
        :type multilabel: bool
        :param prep: enables input preprocessing (default: True)
        :type prep: bool
        :param leave_pbar: controls whether to leave the progress bar or
                           remove it after finishing.
        :type leave_pbar: bool
        :returns: if ``labels`` is True, the list of category names,
                  otherwise, the list of category indexes.
        :rtype: list (of int or str)
        :raises: EmptyModelError, InvalidCategoryError
        """
        if not self.__categories__:
            raise EmptyModelError

        if not def_cat or def_cat == STR_UNKNOWN:
            Print.info(
                "default category was set to 'unknown' (its index will be -1)",
                offset=1
            )
        else:
            if def_cat == STR_MOST_PROBABLE:
                Print.info(
                    "default category was automatically set to '%s' "
                    "(the most probable one)" % self.get_most_probable_category(),
                    offset=1
                )
            else:
                Print.info("default category was set to '%s'" % def_cat, offset=1)
                if self.get_category_index(def_cat) == IDX_UNKNOWN_CATEGORY:
                    raise InvalidCategoryError

        stime = time()
        Print.info("about to start classifying test documents", offset=1)
        classify = self.classify_label if not multilabel else self.classify_multilabel

        y_pred = [
            classify(doc, def_cat=def_cat, labels=labels, prep=prep)
            for doc in tqdm(x_test, desc=" Classification",
                            leave=leave_pbar, disable=Print.is_quiet())
        ]

        Print.info("finished --time: %.1fs" % (time() - stime), offset=1)
        return y_pred


class EmptyModelError(Exception):
    """Exception to be thrown when the model is empty."""

    def __init__(self, msg=''):
        """Class constructor."""
        Exception.__init__(
            self,
            "The model is empty (it hasn't been trained yet)."
        )


class InvalidCategoryError(Exception):
    """Exception to be thrown when a category is not valid."""

    def __init__(self, msg=''):
        """Class constructor."""
        Exception.__init__(
            self,
            "The given category is not valid"
        )


def kmean_multilabel_size(res):
    """
    Use k-means to tell where to split the ``SS3.classify'''s output.

    Given a ``SS3.classify``'s output (``res``), tell where to partition it
    into 2 clusters so that one of the cluster holds the category labels that
    the classifier should output when performing multi-label classification.
    To achieve this, implement k-means (i.e. 2-means) clustering over the
    category confidence values in ``res``.

    :param res: the classification output of ``SS3.classify``
    :type res: list (of sorted pairs (category, confidence value))
    :returns: a positive integer indicating where to split ``res``
    :rtype: int
    """
    cent = {"neg": -1, "pos": -1}  # centroids (2 clusters: "pos" and "neg")
    clust = {"neg": [], "pos": []}  # clusters (2 clusters: "pos" and "neg")
    new_cent_neg = res[-1][1]
    new_cent_pos = res[0][1]
    while (cent["pos"] != new_cent_pos) or (cent["neg"] != new_cent_neg):
        cent["neg"], cent["pos"] = new_cent_neg, new_cent_pos
        clust["neg"], clust["pos"] = [], []
        for _, cat_cv in res:
            if abs(cent["neg"] - cat_cv) < abs(cent["pos"] - cat_cv):
                clust["neg"].append(cat_cv)
            else:
                clust["pos"].append(cat_cv)
        new_cent_neg = sum(clust["neg"]) / len(clust["neg"])
        new_cent_pos = sum(clust["pos"]) / len(clust["pos"])
    return len(clust["pos"])


def sigmoid(v, l):
    """A sigmoid function."""
    try:
        return .5 * tanh((3. / l) * v - 3) + .5
    except ZeroDivisionError:
        return 0


def mad(values, n):
    """Median absolute deviation mean."""
    if len(values) < n:
        values += [0] * int(n - len(values))
    values.sort()
    if n == 2:
        return (values[0], values[0])
    values_m = n // 2 if n % 2 else n // 2 - 1
    m = values[values_m]  # Median
    sd = sum([abs(m - lv) for lv in values]) / float(n)  # sd Mean
    return m, sd


def key_as_int(dct):
    """Cast the given dictionary (numerical) keys to int."""
    keys = list(dct)
    if len(keys) and keys[0].isdigit():
        new_dct = {}
        for key in dct:
            new_dct[int(key)] = dct[key]
        return new_dct
    return dct


def re_split_keep(regex, string):
    """
    Force the inclusion of unmatched items by re.split.

    This allows keeping the original content after splitting the input
    document for later use (e.g. for using it from the Live Test)
    """
    if not re.match(r"\(.*\)", regex):
        regex = "(%s)" % regex
    return re.split(regex, string)


def vsum(v0, v1):
    """Vectorial version of sum."""
    return [v0[i] + v1[i] for i in xrange(len(v0))]


def vmax(v0, v1):
    """Vectorial version of max."""
    return [max(v0[i], v1[i]) for i in xrange(len(v0))]


def vdiv(v0, v1):
    """Vectorial version of division."""
    return [v0[i] / v1[i] if v1[i] else 0 for i in xrange(len(v0))]


def set_verbosity(level):
    """
    Set the verbosity level.

        - ``0`` (quiet): do not output any message (only error messages)
        - ``1`` (normal): default behavior, display only warning messages and progress bars
        - ``2`` (verbose): display also the informative non-essential messages

    The following built-in constants can also be used to refer to these 3 values:
    ``VERBOSITY.QUIET``, ``VERBOSITY.NORMAL``, and ``VERBOSITY.VERBOSE``, respectively.

    For example, if you want PySS3 to hide everything, even progress bars, you could simply do:

    >>> import pyss3
    ...
    >>> pyss3.set_verbosity(0)
    ...
    >>> # here's the rest of your code :D

    or, equivalently:

    >>> import pyss3
    >>> from pyss3 import VERBOSITY
    ...
    >>> pyss3.set_verbosity(VERBOSITY.QUIET)
    ...
    >>> # here's the rest of your code :D

    :param level: the verbosity level
    :type level: int
    """
    Print.set_verbosity(level)


# aliases
SS3.set_smoothness = SS3.set_s
SS3.get_smoothness = SS3.get_s
SS3.set_significance = SS3.set_l
SS3.get_significance = SS3.get_l
SS3.set_sanction = SS3.set_p
SS3.get_sanction = SS3.get_p
SS3.set_alpha = SS3.set_a
SS3.get_alpha = SS3.get_a
