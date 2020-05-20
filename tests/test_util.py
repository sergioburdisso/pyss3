# -*- coding: utf-8 -*-
"""Tests for pyss3.util."""
from os import path
from pyss3 import SS3, InvalidCategoryError
from shutil import rmtree
from pyss3.util import Dataset, Evaluation, RecursiveDefaultDict, Print, VERBOSITY

import sys
import pytest
import pyss3

PY3 = sys.version_info[0] >= 3
DATASET_FOLDER = "dataset_mr"
DATASET_MULTILABEL_FOLDER = "dataset_ml"
DATASET_PATH = path.join(path.abspath(path.dirname(__file__)), DATASET_FOLDER)
DATASET_MULTILABEL_PATH = path.join(path.abspath(path.dirname(__file__)), DATASET_MULTILABEL_FOLDER)
TMP_FOLDER = "tests/ss3_models/"


def test_util():
    """Test utility module."""
    rd = RecursiveDefaultDict()
    rd["a"]["new"]["element"] = "assigned"

    Print.set_verbosity(VERBOSITY.VERBOSE)
    Print.verbosity_region_begin(VERBOSITY.VERBOSE)

    print(Print.style.header("this is a header!"))

    Print.warn("This is a warning!")
    Print.info("This is an informative message!")
    Print.show("This is a message!")

    with pytest.raises(Exception):
        Print.warn("This is a warning!", raises=Exception)

    Print.set_decorator_info(">", "<")
    Print.set_decorator_warn("|", "|")
    Print.set_decorator_error("*", "*")

    Print.verbosity_region_end()


def test_evaluation(mocker):
    """Test Evaluation class."""
    mocker.patch("webbrowser.open")
    mocker.patch("matplotlib.pyplot.show")

    kfold_validation = Evaluation.kfold_cross_validation

    Evaluation.__cache__ = None
    Evaluation.__cache_file__ = None
    Evaluation.__clf__ = None
    Evaluation.__last_eval_tag__ = None
    Evaluation.__last_eval_method__ = None
    Evaluation.__last_eval_def_cat__ = None

    ss = [0, 0.5]
    ll = [0, 1.5]
    pp = [0, 2]
    x_data, y_data = Dataset.load_from_files(DATASET_PATH)
    x_data_ml, y_data_ml = Dataset.load_from_files_multilabel(
        path.join(DATASET_MULTILABEL_PATH, "train/docs.txt"),
        path.join(DATASET_MULTILABEL_PATH, "train/labels.txt"),
        sep_label=",",
        sep_doc="\n>>>>>\n"
    )

    clf = SS3()
    clf.set_model_path("tests")

    clf_ml = SS3(name="multilabel")
    clf_ml.set_model_path("tests")

    # no classifier assigned case
    Evaluation.clear_cache()
    with pytest.raises(ValueError):
        Evaluation.get_best_hyperparameters()
    with pytest.raises(ValueError):
        Evaluation.remove()
    with pytest.raises(ValueError):
        Evaluation.show_best()
    with pytest.raises(ValueError):
        Evaluation.plot(TMP_FOLDER)

    # Not-yet-trained model case
    Evaluation.set_classifier(clf)
    Evaluation.clear_cache()
    Evaluation.remove()
    Evaluation.show_best()
    assert Evaluation.plot(TMP_FOLDER) is False

    with pytest.raises(pyss3.EmptyModelError):
        Evaluation.test(clf, x_data, y_data)
    with pytest.raises(pyss3.EmptyModelError):
        kfold_validation(clf, x_data, y_data)
    with pytest.raises(pyss3.EmptyModelError):
        Evaluation.grid_search(clf, x_data, y_data)
    with pytest.raises(LookupError):
        Evaluation.get_best_hyperparameters()

    # default argument values
    clf.train(x_data, y_data)
    clf_ml.train(x_data_ml, y_data_ml)

    assert Evaluation.test(clf, x_data, y_data, plot=PY3) == 1
    assert Evaluation.test(clf, ['bla bla bla'], ['pos'], plot=PY3) == 0
    assert Evaluation.test(clf,
                           ['bla bla bla', "I love this love movie!"],
                           ['pos', 'pos'],
                           plot=PY3) == 0.5
    assert kfold_validation(clf_ml, x_data_ml, y_data_ml, plot=PY3) > 0
    assert kfold_validation(clf, x_data, y_data, plot=PY3) > 0
    s, l, p, a = clf.get_hyperparameters()
    s, l, p, a = clf.get_hyperparameters()
    s0, l0, p0, a0 = Evaluation.grid_search(clf, x_data, y_data)
    s1, l1, p1, a1 = Evaluation.get_best_hyperparameters()
    s2, l2, p2, a2 = Evaluation.get_best_hyperparameters("recall")
    Evaluation.__last_eval_tag__ = None
    s1, l1, p1, a1 = Evaluation.get_best_hyperparameters()
    assert s0 == s and l0 == l and p0 == p and a0 == a
    assert s0 == s1 and l0 == l1 and p0 == p1 and a0 == a1
    assert s0 == s2 and l0 == l2 and p0 == p2 and a0 == a2
    assert Evaluation.plot(TMP_FOLDER) is True
    Evaluation.remove()
    Evaluation.show_best()
    assert Evaluation.plot(TMP_FOLDER) is False

    # test
    #   OK
    assert Evaluation.test(clf_ml, x_data_ml, y_data_ml, plot=PY3) == .3125
    assert Evaluation.test(clf_ml, x_data_ml, y_data_ml, metric='exact-match', plot=PY3) == .3
    assert Evaluation.test(clf, x_data, y_data, def_cat='unknown', plot=PY3) == 1
    assert Evaluation.test(clf, x_data, y_data, def_cat='neg', plot=PY3) == 1
    assert Evaluation.test(clf, x_data, y_data, metric="f1-score", plot=PY3) == 1
    assert Evaluation.test(clf, x_data, y_data, plot=PY3,
                           metric="recall", metric_target="weighted avg") == 1
    assert Evaluation.test(clf, x_data, y_data, plot=PY3,
                           metric="recall", metric_target="neg") == 1
    #   Not OK
    with pytest.raises(InvalidCategoryError):
        Evaluation.test(clf, x_data, y_data, def_cat='xxx', plot=PY3)
    with pytest.raises(KeyError):
        Evaluation.test(clf, x_data, y_data, metric="xxx", plot=PY3)
    with pytest.raises(KeyError):
        Evaluation.test(clf, x_data, y_data, metric="recall", metric_target="xxx", plot=PY3)
    with pytest.raises(ValueError):
        Evaluation.test(clf, x_data, y_data, metric='hamming-loss')

    # k-fold
    #   OK
    assert kfold_validation(clf, x_data, y_data, n_grams=3, plot=PY3) > 0
    assert kfold_validation(clf, x_data, y_data, k=10, plot=PY3) > 0
    assert kfold_validation(clf, x_data, y_data, k=10, def_cat='unknown', plot=PY3) > 0
    assert kfold_validation(clf, x_data, y_data, k=10, def_cat='neg', plot=PY3) > 0
    assert kfold_validation(clf, x_data, y_data, metric="f1-score", plot=PY3) > 0
    assert kfold_validation(clf, x_data, y_data, plot=PY3,
                            metric="recall", metric_target="weighted avg") > 0
    assert kfold_validation(clf, x_data, y_data, plot=PY3,
                            metric="recall", metric_target="neg") > 0
    #   Not OK
    with pytest.raises(ValueError):
        kfold_validation(clf, x_data, y_data, n_grams=-1, plot=PY3)
    with pytest.raises(ValueError):
        kfold_validation(clf, x_data, y_data, n_grams=clf, plot=PY3)
    with pytest.raises(ValueError):
        kfold_validation(clf, x_data, y_data, k=-1, plot=PY3)
    with pytest.raises(ValueError):
        kfold_validation(clf, x_data, y_data, k=clf, plot=PY3)
    with pytest.raises(ValueError):
        kfold_validation(clf, x_data, y_data, k=None, plot=PY3)
    with pytest.raises(InvalidCategoryError):
        kfold_validation(clf, x_data, y_data, def_cat='xxx', plot=PY3)
    with pytest.raises(KeyError):
        kfold_validation(clf, x_data, y_data, metric="xxx", plot=PY3)
    with pytest.raises(KeyError):
        kfold_validation(clf, x_data, y_data, metric="recall", metric_target="xxx", plot=PY3)

    # grid_search
    #   OK
    s0, l0, p0, a0 = Evaluation.grid_search(clf, x_data, y_data, s=ss)
    s1, l1, p1, a1 = Evaluation.grid_search(clf_ml, x_data_ml, y_data_ml, s=ss, l=ll, p=pp)
    s1, l1, p1, a1 = Evaluation.grid_search(clf, x_data, y_data, s=ss, l=ll, p=pp)
    assert s0 == s1 and l0 == l1 and p0 == p1 and a0 == a1
    s0, l0, p0, a0 = Evaluation.grid_search(clf_ml, x_data_ml, y_data_ml, k_fold=4)
    s0, l0, p0, a0 = Evaluation.grid_search(clf, x_data, y_data, k_fold=4)
    s0, l0, p0, a0 = Evaluation.grid_search(clf, x_data, y_data, def_cat='unknown', p=pp)
    s1, l1, p1, a1 = Evaluation.grid_search(clf, x_data, y_data, def_cat='neg', p=pp)
    assert s0 == s1 and l0 == l1 and p0 == p1 and a0 == a1
    s0, l0, p0, a0 = Evaluation.grid_search(clf, x_data, y_data, metric="f1-score", p=pp)
    s1, l1, p1, a1 = Evaluation.grid_search(clf, x_data, y_data, p=pp,
                                            metric="recall", metric_target="weighted avg")
    s1, l1, p1, a1 = Evaluation.grid_search(clf, x_data, y_data, p=pp,
                                            metric="recall", metric_target="neg")
    assert s0 == s1 and l0 == l1 and p0 == p1 and a0 == a1
    #   Not OK
    with pytest.raises(TypeError):
        Evaluation.grid_search(clf, x_data, y_data, s='asd')
    with pytest.raises(TypeError):
        Evaluation.grid_search(clf, x_data, y_data, s=clf)
    with pytest.raises(TypeError):
        Evaluation.grid_search(clf, x_data, y_data, k_fold=clf)
    with pytest.raises(TypeError):
        Evaluation.grid_search(clf, x_data, y_data, k_fold="xxx")
    with pytest.raises(InvalidCategoryError):
        Evaluation.grid_search(clf, x_data, y_data, def_cat='xxx')
    with pytest.raises(KeyError):
        Evaluation.grid_search(clf, x_data, y_data, metric="xxx")
    with pytest.raises(KeyError):
        Evaluation.grid_search(clf, x_data, y_data, metric="recall", metric_target="xxx")

    # get_best_hyperparameters
    s1, l1, p1, a1 = Evaluation.get_best_hyperparameters()
    s2, l2, p2, a2 = Evaluation.get_best_hyperparameters("recall")
    s1, l1, p1, a1 = Evaluation.get_best_hyperparameters("recall", "weighted avg")
    s1, l1, p1, a1 = Evaluation.get_best_hyperparameters("recall", "pos")
    s1, l1, p1, a1 = Evaluation.get_best_hyperparameters(method="10-fold")
    s1, l1, p1, a1 = Evaluation.get_best_hyperparameters(method="10-fold", def_cat="neg")
    s1, l1, p1, a1 = Evaluation.get_best_hyperparameters(method="10-fold", def_cat="unknown")
    assert s0 == s1 and l0 == l1 and p0 == p1 and a0 == a1
    assert s0 == s2 and l0 == l2 and p0 == p2 and a0 == a2

    # Not OK
    with pytest.raises(KeyError):
        Evaluation.get_best_hyperparameters("xxx")
    with pytest.raises(KeyError):
        Evaluation.get_best_hyperparameters("recall", "xxx")
    with pytest.raises(LookupError):
        Evaluation.get_best_hyperparameters(method="xxx")
    with pytest.raises(LookupError):
        Evaluation.get_best_hyperparameters(def_cat="xxx")
    with pytest.raises(LookupError):
        Evaluation.get_best_hyperparameters(method="4-fold", def_cat="unknown")

    # plot OK
    assert Evaluation.plot(TMP_FOLDER) is True

    # remove
    #   OK
    assert Evaluation.remove(s, l, p, a)[0] == 10
    assert Evaluation.remove(def_cat="neg")[0] == 2
    assert Evaluation.remove(method="test")[0] == 12
    assert Evaluation.remove(s=-10)[0] == 0
    assert Evaluation.remove(def_cat="xxx")[0] == 0
    assert Evaluation.remove(method="xxx")[0] == 0
    assert Evaluation.remove()[0] == 1
    assert Evaluation.plot(TMP_FOLDER) is False  # plot not OK (no evaluations)
    #   not OK
    with pytest.raises(TypeError):
        Evaluation.remove("xxx")
    with pytest.raises(TypeError):
        Evaluation.remove(clf)

    Evaluation.show_best()
    Evaluation.show_best(method="test")
    Evaluation.show_best(def_cat="unknown")
    Evaluation.show_best(metric="f1-score")
    Evaluation.show_best(metric="f1-score", avg="weighted avg")

    # different tag

    rmtree("./tests/ss3_models", ignore_errors=True)


def test_dataset():
    """Test Dataset class."""
    x_train, y_train = Dataset.load_from_files_multilabel(
        path.join(DATASET_MULTILABEL_PATH, "train_files"),
        path.join(DATASET_MULTILABEL_PATH, "file_labels.tsv")
    )

    assert x_train == ['this is the first document!!\n\n:)', 'and this is the\n\nSECOND!!']
    assert y_train == [['catA', 'catB', 'catC'], ['catA']]

    x_train, y_train = Dataset.load_from_files_multilabel(
        path.join(DATASET_MULTILABEL_PATH, "train/docs.txt"),
        path.join(DATASET_MULTILABEL_PATH, "train/labels.txt"),
        sep_label=",",
        sep_doc="\n>>>>>\n"
    )

    assert len(y_train) == len(y_train) and len(y_train) == 20
    assert y_train[:8] == [[], ['toxic', 'severe_toxic', 'obscene', 'insult'],
                           [], [], [], [], [], ['toxic']]
