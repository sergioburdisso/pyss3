# -*- coding: utf-8 -*-
"""Tests for pyss3."""
from os import path
from shutil import rmtree
from pyss3.util import Dataset, membership_matrix
from pyss3 import \
    SS3, STR_NORM_GV_XAI, STR_NORM_GV, STR_GV, \
    STR_XAI, STR_VANILLA, STR_MOST_PROBABLE, \
    STR_UNKNOWN, STR_UNKNOWN_CATEGORY, IDX_UNKNOWN_CATEGORY, \
    PARA_DELTR, SENT_DELTR, WORD_DELTR, VERBOSITY

import sys
import pyss3
import pytest

pyss3.set_verbosity(VERBOSITY.QUIET)

DATASET_FOLDER = "dataset"
DATASET_MULTILABEL_FOLDER = "dataset_ml"

dataset_path = path.join(path.abspath(path.dirname(__file__)), DATASET_FOLDER)
dataset_multilabel_path = path.join(path.abspath(path.dirname(__file__)), DATASET_MULTILABEL_FOLDER)

x_train, y_train = Dataset.load_from_files(dataset_path, folder_label=False)
x_test = [
    "sports nfl nba superbowl soccer football team. learns jersey air bowl hockey.\n"
    "baseball helmet mccutchen jordan curry poker.",

    "travel pictures images moment glamour canvas photoshoot lens dslr portrait "
    "beautiful seasons lines colours snap usm af eos painter gallery museum  "
    "flower kinkade.",

    "hairstyles boutique handbag dress trends womens menswear luxury claudiepierlot "
    "rustic wedding bride collection signed patrick ista streetstyle cocksox purse "
    "trending status brush cosmetic stretchy gucci leather cream trendy "
    "bargains victoria.",

    "finance business development fiverr hiring job social debt logos stationary "
    "read bad media mlm uganda entrepreneurship strategy mistake 1st employee "
    "financial inbound habits coupon.",

    "cooking cook chef food drink rice kitchen cold organic yummy yum bread "
    "strawberry bbq pepper beverages grocery cupcakes easter gurpreet sushi "
    "dining meal chicken lime mushrooms restaurant whiskey.",

    "vitamins calcium minerals workout weightloss fit skin spa motivation care "
    "health yoga food shampoo niche 100ml juvederm losing edp munched "
    "rejuvenating lipstick vegetables.",

    "rock roll radio entertainment playing song celine dion hiphop sinatra britney "
    "spears nowplaying music flow streaming dalston fm hogan songs taylor.",

    "android mobile ios science nasa space data hp enterprise earth major could dns "
    "virtualization teachers strategic spending distribusion comets virtual universe."
]
y_test = ["sports",
          "art&photography",
          "beauty&fashion",
          "business&finance",
          "food",
          "health",
          "music",
          "science&technology"]

STOPWORDS = ['by', 'the', 'for', 'of', 'new', 'to', 'with', 'is', 'at', 'and', 'in', 'this', 'out']

doc_insight = "Dude, this text is about sports. Football soccer, you know!\n2nd paragraph."
doc_unknown = "bla bla bla."
doc_blocks0 = "is this a sentence? a paragraph?!who knows"
doc_blocks1 = "these-are-words"


@pytest.fixture()
def mockers(mocker):
    """Set mockers up."""
    mocker.patch("matplotlib.pyplot.show")


def argmax(lst):
    """Given a list of numbers, return the index of the biggest one."""
    return max(range(len(lst)), key=lst.__getitem__)


def perform_tests_with(clf, cv_test, stopwords=True):
    """Perform some tests with the given classifier."""
    multilabel_doc = x_test[0] + x_test[1]
    multilabel_labels = [y_test[0], y_test[1]]
    multilabel_idxs = [clf.get_category_index(y_test[0]),
                       clf.get_category_index(y_test[1])]
    new_cat = "bla"
    def_cat = "music"
    def_cat_idx = clf.get_category_index(def_cat)
    most_prob_cat = clf.get_most_probable_category()
    most_prob_cat_idx = clf.__get_most_probable_category__()

    # category names case-sensitiveness check
    assert clf.get_category_index("SpOrTs") == -1

    # predict
    y_pred = clf.predict(x_test)
    assert y_pred == y_test
    clf.set_a(.1)
    y_pred = clf.predict(x_test)
    assert y_pred == y_test
    clf.set_a(0)
    y_pred = clf.predict(x_test, multilabel=True)
    assert y_pred == [[y] for y in y_test]
    y_pred = clf.predict(x_test, multilabel=True, labels=False)
    assert y_pred == [[clf.get_category_index(y)] for y in y_test]

    y_pred = clf.predict(x_test, labels=False)
    y_pred = [clf.get_category_name(ic) for ic in y_pred]
    assert y_pred == y_test

    y_pred = clf.predict([doc_unknown], def_cat=STR_UNKNOWN)
    assert y_pred[0] == STR_UNKNOWN_CATEGORY
    y_pred = clf.predict([doc_unknown], multilabel=True)
    assert y_pred[0] == []
    y_pred = clf.predict([doc_unknown], def_cat=STR_MOST_PROBABLE, multilabel=True, labels=False)
    assert y_pred[0] == [most_prob_cat_idx]

    y_pred = clf.predict([doc_unknown], def_cat=STR_MOST_PROBABLE)
    assert y_pred[0] == most_prob_cat
    assert y_pred[0] == "science&technology"

    assert clf.predict([doc_unknown], def_cat=def_cat)[0] == def_cat

    # predict_proba
    y_pred = clf.predict_proba(x_test)
    assert y_test == [clf.get_category_name(argmax(cv)) for cv in y_pred]
    assert [round(p, 5) for p in y_pred[0]] == cv_test

    y_pred = clf.predict_proba([doc_unknown])
    assert y_pred[0] == [0] * len(clf.get_categories())

    # classify
    assert clf.classify(None)[0][1] == 0
    assert clf.classify("")[0][1] == 0

    pred = clf.classify(doc_unknown, sort=False, prep=False)
    assert pred == [0] * len(clf.get_categories())

    pred = clf.classify(doc_unknown, sort=False)
    assert pred == [0] * len(clf.get_categories())

    pred0 = clf.classify(x_test[0], sort=False)
    assert argmax(pred0) == clf.get_category_index(y_test[0])

    pred1 = clf.classify(x_test[0], sort=True)
    assert pred1[0][0] == clf.get_category_index(y_test[0])
    assert argmax(pred0) == pred1[0][0] and pred0[argmax(pred0)] == pred1[0][1]

    # classify_label
    assert clf.classify_label(x_test[0]) == y_test[0]
    assert clf.classify_label(x_test[0], labels=False) == clf.get_category_index(y_test[0])

    assert clf.classify_label('') == most_prob_cat
    assert clf.classify_label('', def_cat=STR_UNKNOWN) == STR_UNKNOWN_CATEGORY
    assert clf.classify_label('', def_cat=def_cat) == def_cat

    assert clf.classify_label(doc_unknown) == most_prob_cat
    assert clf.classify_label(doc_unknown, def_cat=STR_UNKNOWN) == STR_UNKNOWN_CATEGORY
    assert clf.classify_label(doc_unknown, def_cat=def_cat) == def_cat

    assert clf.classify_label(doc_unknown, labels=False) == most_prob_cat_idx
    assert clf.classify_label(doc_unknown, def_cat=STR_UNKNOWN, labels=False) == -1
    assert clf.classify_label(doc_unknown, def_cat=def_cat, labels=False) == def_cat_idx

    # classify_multilabel

    r = clf.classify_multilabel(multilabel_doc)
    assert len(multilabel_labels) == len(r)
    assert r[0] in multilabel_labels and r[1] in multilabel_labels
    r = clf.classify_multilabel(multilabel_doc, labels=False)
    assert len(multilabel_labels) == len(r)
    assert r[0] in multilabel_idxs and r[1] in multilabel_idxs

    assert clf.classify_multilabel('') == []
    assert clf.classify_multilabel('', def_cat=STR_MOST_PROBABLE) == [most_prob_cat]
    assert clf.classify_multilabel('', def_cat=def_cat) == [def_cat]

    assert clf.classify_multilabel(doc_unknown) == []
    assert clf.classify_multilabel(doc_unknown, def_cat=STR_MOST_PROBABLE) == [most_prob_cat]
    assert clf.classify_multilabel(doc_unknown, def_cat=def_cat) == [def_cat]

    assert clf.classify_multilabel(doc_unknown, labels=False) == []
    assert clf.classify_multilabel(doc_unknown, def_cat=STR_MOST_PROBABLE,
                                   labels=False) == [most_prob_cat_idx]
    assert clf.classify_multilabel(doc_unknown, def_cat=def_cat, labels=False) == [def_cat_idx]

    # "learn an doc_unknown and a new category" case
    clf.learn(doc_unknown * 2, new_cat, update=True)
    assert new_cat in clf.get_categories()
    y_pred = clf.predict([doc_unknown])
    assert y_pred[0] == new_cat

    # get_stopwords
    if stopwords:
        learned_stopwords = clf.get_stopwords(.01)
        assert [sw for sw in STOPWORDS if sw in learned_stopwords] == STOPWORDS

    # set_block_delimiters
    pred = clf.classify(doc_blocks0, json=True)
    assert len(pred["pars"]) == 1 and len(pred["pars"][0]["sents"]) == 1
    assert len(pred["pars"][0]["sents"][0]["words"]) == 15  # spaces and symbols included

    clf.set_block_delimiters(parag="!", sent=r"\?")
    pred = clf.classify(doc_blocks0, json=True)
    assert len(pred["pars"]) == 2 + 1  # 2 paragraphs + 1 delimiter
    assert len(pred["pars"][0]["sents"]) == 4
    clf.set_block_delimiters(sent=r"(\?)")
    assert len(pred["pars"][0]["sents"]) == 4

    clf.set_block_delimiters(word="-")
    pred = clf.classify(doc_blocks1, json=True)
    assert len(pred["pars"][0]["sents"][0]["words"]) == 5  # 3 words + 2 delimiters
    clf.set_block_delimiters(parag=PARA_DELTR, sent=SENT_DELTR, word=WORD_DELTR)


def perform_tests_on(fn, value, ngram="chicken", cat="food"):
    """Perform tests on gv, lv, sn, or sg."""
    assert round(fn(ngram, cat), 4) == value
    assert round(fn("xxx", cat), 4) == 0
    assert round(fn("the xxx chicken", cat), 4) == 0
    assert round(fn("", cat), 4) == 0
    with pytest.raises(pyss3.InvalidCategoryError):
        fn("chicken", "xxx")
    with pytest.raises(pyss3.InvalidCategoryError):
        fn("chicken", "")
    with pytest.raises(pyss3.InvalidCategoryError):
        fn("", "")


def test_pyss3_functions():
    """Test pyss3 functions."""
    assert pyss3.sigmoid(1, 0) == 0
    assert pyss3.sigmoid(1, 1) == .5
    assert pyss3.sigmoid(.2, .2) == .5
    assert pyss3.sigmoid(.5, .5) == .5
    assert round(pyss3.sigmoid(0, .5), 5) == .00247
    assert round(pyss3.sigmoid(1, .5), 5) == .99753
    assert round(pyss3.sigmoid(1, 2), 5) == .04743

    assert pyss3.mad([1, 1, 1], 3) == (1, .0)
    assert pyss3.mad([1, 1, 1], 3) == (1, .0)
    assert pyss3.mad([], 1) == (0, .0)
    assert round(pyss3.mad([1, 2, 1], 3)[1], 5) == .33333
    assert round(pyss3.mad([1, 10, 1], 3)[1], 5) == 3.0

    r = [(6, 8.1), (7, 5.6), (2, 5.5), (4, 1.5),
         (5, 1.3), (3, 1.2), (0, 1.1), (1, 0.4)]
    assert pyss3.kmean_multilabel_size(r) == 3
    assert pyss3.kmean_multilabel_size([(0, 0), (1, 0)]) == 0
    assert pyss3.kmean_multilabel_size([(0, 10), (1, 10)]) == 0

    with pytest.raises(IndexError):
        pyss3.mad([], 0)


def test_multilabel():
    """Test multilabel support."""
    x_train, y_train = Dataset.load_from_files_multilabel(
        path.join(dataset_multilabel_path, "train/docs.txt"),
        path.join(dataset_multilabel_path, "train/labels.txt"),
        sep_label=",",
        sep_doc="\n>>>>>\n"
    )

    clf = SS3()

    with pytest.raises(ValueError):
        membership_matrix(clf, [])

    clf.fit(x_train, y_train)

    assert sorted(clf.get_categories()) == ['insult', 'obscene', 'severe_toxic', 'toxic']
    assert clf.classify_multilabel("this is a unknown document!") == []

    y_pred = [[], ['toxic'], ['severe_toxic'], ['obscene'], ['insult'], ['toxic', 'insult']]

    y_pred_memmatrix = membership_matrix(clf, y_pred).todense().tolist()
    assert y_pred_memmatrix == [[0, 0, 0, 0],  # []
                                [1, 0, 0, 0],  # ['toxic']
                                [0, 1, 0, 0],  # ['severe_toxic']
                                [0, 0, 1, 0],  # ['obscene']
                                [0, 0, 0, 1],  # ['insult']
                                [1, 0, 0, 1]]  # ['toxic', 'insult']

    y_pred_memmatrix = membership_matrix(clf, y_pred + [["xxx"]]).todense().tolist()
    assert y_pred_memmatrix[-1] == [0, 0, 0, 0]


def test_pyss3_ss3(mockers):
    """Test SS3."""
    with pytest.raises(ValueError):
        clf = SS3("hyperparameter")

    clf = SS3(
        s=.45, l=.5, p=1, a=0,
        cv_m=STR_NORM_GV_XAI, sn_m=STR_XAI
    )
    clf.set_name("test")

    # "cold start" tests
    assert clf.get_name() == "test"
    assert clf.get_category_index("a_category") == IDX_UNKNOWN_CATEGORY
    assert clf.get_category_name(0) == STR_UNKNOWN_CATEGORY
    assert clf.get_category_name(-1) == STR_UNKNOWN_CATEGORY
    assert clf.get_ngrams_length() == 0

    with pytest.raises(pyss3.EmptyModelError):
        clf.classify(x_test[0])
    with pytest.raises(pyss3.EmptyModelError):
        clf.predict(x_test)
    with pytest.raises(pyss3.EmptyModelError):
        clf.predict_proba(x_test)
    with pytest.raises(ValueError):
        clf.train(x_train, [])
    with pytest.raises(ValueError):
        clf.train([], [])

    # train and predict/classify tests (model: terms are single words)
    # cv_m=STR_NORM_GV_XAI, sn_m=STR_XAI
    clf.fit(x_train, y_train)

    assert clf.get_ngrams_length() == 1

    perform_tests_with(clf, [.00114, .00294, 0, 0, 0, .00016, .01878, 8.43969])
    perform_tests_on(clf.cv, 0.4307)
    perform_tests_on(clf.gv, 0.2148)
    perform_tests_on(clf.lv, 0.2148)
    perform_tests_on(clf.sg, 1)
    perform_tests_on(clf.sn, 1)
    perform_tests_on(clf.cv, 0, "video games", "science&technology")
    perform_tests_on(clf.gv, 0, "video games", "science&technology")

    # cv_m=STR_NORM_GV, sn_m=STR_XAI
    clf = SS3(
        s=.45, l=.5, p=1, a=0, name="test-norm-gv-sn-xai",
        cv_m=STR_NORM_GV, sn_m=STR_XAI
    )
    clf.fit(x_train, y_train)

    perform_tests_with(clf, [.00114, .00294, 0, 0, 0, .00016, .01878, 8.43969])
    perform_tests_on(clf.cv, 0.4307)

    # cv_m=STR_GV, sn_m=STR_XAI
    clf = SS3(
        s=.45, l=.5, p=1, a=0, name="test-gv-sn-xai",
        cv_m=STR_GV, sn_m=STR_XAI
    )
    clf.fit(x_train, y_train)

    perform_tests_with(clf, [.00062, .00109, 0, 0, 0, .00014, .01878, 6.31605])
    assert clf.cv("chicken", "food") == clf.gv("chicken", "food")

    # cv_m=STR_NORM_GV_XAI, sn_m=STR_VANILLA
    clf = SS3(
        s=.45, l=.5, p=1, a=0, name="test-norm-gv-xai-sn-vanilla",
        cv_m=STR_NORM_GV_XAI, sn_m=STR_VANILLA
    )
    clf.fit(x_train, y_train)

    perform_tests_with(clf, [.00114, .00294, 0, 0, 0, .00016, .01878, 8.43969], stopwords=False)

    # train and predict/classify tests (model: terms are word n-grams)
    clf = SS3(
        name="test-3grams",
        cv_m=STR_NORM_GV_XAI, sn_m=STR_XAI
    )

    clf.fit(x_train, y_train, n_grams=3)

    assert clf.get_ngrams_length() == 3

    # update_values
    clf.set_l(.3)
    clf.update_values()
    clf.set_p(.2)
    clf.update_values()
    clf.set_hyperparameters(s=.32, l=1.24, p=1.1, a=0)
    clf.update_values()
    clf.update_values()

    perform_tests_with(clf, [.00074, .00124, 0, 0, 0, .00028, .00202, 9.17139])
    perform_tests_on(clf.cv, 1.5681, "video games", "science&technology")
    perform_tests_on(clf.gv, 0.6697, "video games", "science&technology")
    perform_tests_on(clf.lv, 0.6697, "video games", "science&technology")
    perform_tests_on(clf.sg, 1, "video games", "science&technology")
    perform_tests_on(clf.sn, 1, "video games", "science&technology")

    # n-gram recognition tests
    pred = clf.classify("android mobile and video games", json=True)
    assert pred["pars"][0]["sents"][0]["words"][0]["lexeme"] == "android mobile"
    assert pred["pars"][0]["sents"][0]["words"][-1]["lexeme"] == "video games"
    assert argmax(pred["cv"]) == clf.get_category_index("science&technology")
    assert [round(p, 5) for p in pred["cv"]] == [0, 0, 0, 0, 0, 0, 4.38069, 0, 0]

    pred = clf.classify("playing football soccer", json=True)
    assert pred["pars"][0]["sents"][0]["words"][-1]["lexeme"] == "football soccer"
    assert argmax(pred["cv"]) == clf.get_category_index("sports")
    assert [round(p, 5) for p in pred["cv"]] == [0, 0, 0, 0, 0, .53463, 0, 1.86708, 0]

    # extract_insight
    t = clf.extract_insight(doc_insight)
    assert len(t) == 1 and t[0][0] == ' about sports. Football soccer, you know!'
    t = clf.extract_insight(doc_insight, window_size=1)
    assert len(t) == 1 and t[0] == (' sports. Football soccer, you know', 2.8670788645841605)
    t = clf.extract_insight(doc_insight, window_size=0)
    assert t == [('Football soccer', 1.8670788645841605), ('sports', 1.0)]
    assert clf.extract_insight(doc_insight, cat="music") == []
    assert len(clf.extract_insight(doc_insight, min_cv=1)) == 1
    t = clf.extract_insight(doc_insight, level="sentence")
    assert len(t) == 2 and t[0][0] == ' Football soccer, you know!'
    t = clf.extract_insight(doc_insight, level="sentence", sort=False)
    assert len(t) == 2 and t[0][0] == 'Dude, this text is about sports'
    t = clf.extract_insight(doc_insight, level="paragraph", min_cv=-1)
    assert len(t) == 2 + 1 and t[2][0] == "2nd paragraph."

    with pytest.raises(pyss3.InvalidCategoryError):
        clf.extract_insight(doc_insight, cat=STR_UNKNOWN_CATEGORY)
    with pytest.raises(ValueError):
        clf.extract_insight(doc_insight, level="invalid")

    # prints
    clf.print_model_info()
    clf.print_hyperparameters_info()
    clf.print_categories_info()
    clf.print_ngram_info("video games")

    # plot_value_distribution
    if sys.version_info[0] >= 3:
        clf.plot_value_distribution(y_test[0])

    # load and save model tests
    clf.set_model_path("tests/")
    clf.save_model()
    clf.load_model()

    # get_next_words
    assert clf.get_next_words("android", "science&technology")[0][0] == "mobile"

    clf = SS3(name="test-3grams")

    with pytest.raises((OSError, IOError)):
        clf.set_model_path("dummy")
        clf.load_model()

    clf.set_model_path("./tests")
    clf.load_model()

    clf.set_model_path("tests/tmp")
    clf.save_model()
    clf.save_model()
    clf.load_model()

    clf.save_model("tests/")
    clf.load_model()

    clf = SS3(name="test-3grams")
    clf.load_model("./tests/")

    clf.save_model("./tests/tmp/")
    clf.save_model()
    clf.load_model()

    # save_vocab
    clf.save_vocab("tests/tmp")

    # save_cat_vocab
    clf.save_cat_vocab("sports", "tests/tmp")
    with pytest.raises(pyss3.InvalidCategoryError):
        clf.save_cat_vocab(STR_UNKNOWN_CATEGORY, "tests/tmp")

    rmtree("./tests/tmp", ignore_errors=True)
    rmtree("./tests/ss3_models", ignore_errors=True)


# if __name__ == "__main__":
#     test_pyss3_ss3()
