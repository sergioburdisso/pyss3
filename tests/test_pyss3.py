"""Tests for pytest."""
from os import path
from shutil import rmtree
from pyss3 import SS3, STR_NORM_GV_XAI, STR_XAI
from pyss3 import STR_UNKNOWN, STR_MOST_PROBABLE, STR_UNKNOWN_CATEGORY, IDX_UNKNOWN_CATEGORY
from pyss3.util import Dataset
import pyss3
import pytest

DATASET_FOLDER = "dataset"

dataset_path = path.join(path.abspath(path.dirname(__file__)), DATASET_FOLDER)

x_train, y_train = Dataset.load_from_files(dataset_path, folder_label=False)
x_test = [
    "sports nfl nba superbowl soccer football team. learns jersey air bowl hockey.\n"
    "baseball helmet mccutchen jordan curry poker",

    "travel pictures images moment glamour canvas photoshoot lens dslr portrait "
    "beautiful seasons lines colours snap usm af eos painter gallery museum  "
    "flower kinkade",

    "hairstyles boutique handbag dress trends womens menswear luxury claudiepierlot "
    "rustic wedding bride collection signed patrick ista streetstyle cocksox purse "
    "trending status brush cosmetic stretchy gucci leather cream trendy "
    "bargains victoria",

    "finance business development fiverr hiring job social debt logos stationary "
    "read bad media mlm uganda entrepreneurship strategy mistake 1st employee "
    "financial inbound habits coupon",

    "cooking cook chef food drink rice kitchen cold organic yummy yum bread "
    "strawberry bbq pepper beverages grocery cupcakes easter gurpreet sushi "
    "dining meal chicken lime mushrooms restaurant whiskey",

    "vitamins calcium minerals workout weightloss fit skin spa motivation care "
    "health yoga food shampoo niche 100ml juvederm losing edp munched "
    "rejuvenating lipstick vegetables",

    "rock roll radio entertainment playing song celine dion hiphop sinatra britney "
    "spears nowplaying music flow streaming dalston fm hogan songs taylor",

    "android mobile ios science nasa space data hp enterprise earth major could dns "
    "virtualization teachers strategic spending distribusion comets virtual universe"
]
y_test = ["sports",
          "art&photography",
          "beauty&fashion",
          "business&finance",
          "food",
          "health",
          "music",
          "science&technology"]

stopwords = ['by', 'the', 'for', 'of', 'new', 'to', 'with', 'is', 'at', 'and', 'in', 'this', 'out']


def argmax(lst):
    """Given a list of numbers, return the index of the biggest one."""
    return max(range(len(lst)), key=lst.__getitem__)


def perform_tests_with(clf, cv_test):
    """Perform some tests with the given classifier."""
    assert clf.get_category_index("SpOrTs") == clf.get_category_index("sports")

    y_pred = clf.predict(x_test)
    assert y_pred == y_test

    y_pred = clf.predict(x_test, labels=False)
    y_pred = [clf.get_category_name(ic) for ic in y_pred]
    assert y_pred == y_test

    y_pred = clf.predict(["bla bla bla"], def_cat=STR_UNKNOWN)
    assert y_pred[0] == STR_UNKNOWN_CATEGORY

    y_pred = clf.predict(["bla bla bla"], def_cat=STR_MOST_PROBABLE)
    assert y_pred[0] == clf.get_most_probable_category()
    assert y_pred[0] == "science&technology"

    y_pred = clf.predict(["bla bla bla"], def_cat="music")
    assert y_pred[0] == "music"

    y_pred = clf.predict_proba(x_test)
    assert y_test == [clf.get_category_name(argmax(cv)) for cv in y_pred]
    assert [round(p, 5) for p in y_pred[0]] == cv_test

    y_pred = clf.predict_proba(["bla bla bla"])
    assert y_pred[0] == [0] * len(clf.get_categories())

    pred = clf.classify("bla bla bla", sort=False)
    assert pred == [0] * len(clf.get_categories())

    pred0 = clf.classify(x_test[0], sort=False)
    assert argmax(pred0) == clf.get_category_index(y_test[0])

    pred1 = clf.classify(x_test[0], sort=True)
    assert pred1[0][0] == clf.get_category_index(y_test[0])

    assert argmax(pred0) == pred1[0][0] and pred0[argmax(pred0)] == pred1[0][1]

    clf.learn("bla bla bla bla bla bla", "bla", update=True)

    assert "bla" in clf.get_categories()

    y_pred = clf.predict(["bla bla bla"])
    assert y_pred[0] == "bla"

    learned_stopwords = clf.get_stopwords(.01)
    assert [sw for sw in stopwords if sw in learned_stopwords] == stopwords


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

    with pytest.raises(IndexError):
        pyss3.mad([], 0)


def test_pyss3_ss3():
    """Test SS3."""
    clf = SS3(
        s=.45, l=.5, p=1, a=0, name="test",
        cv_m=STR_NORM_GV_XAI, sn_m=STR_XAI
    )
    assert clf.get_name() == "test"
    assert clf.get_category_index("a_category") == IDX_UNKNOWN_CATEGORY
    assert clf.get_category_name(0) == STR_UNKNOWN_CATEGORY
    assert clf.get_category_name(-1) == STR_UNKNOWN_CATEGORY

    with pytest.raises(pyss3.EmptyModelError):
        y_pred = clf.predict(x_test)
    with pytest.raises(pyss3.EmptyModelError):
        y_pred = clf.predict_proba(x_test)

    clf.fit(x_train, y_train)

    perform_tests_with(clf, [.00114, .00295, 0, 0, 0, .00016, .01894, 8.47741])

    clf = SS3(
        s=.32, l=1.24, p=1.1, a=0, name="test-3grams",
        cv_m=STR_NORM_GV_XAI, sn_m=STR_XAI
    )
    clf.fit(x_train, y_train, n_grams=3)

    perform_tests_with(clf, [.00074, .00124, 0, 0, 0, .00028, .00202, 9.19105])

    pred = clf.classify("android mobile and video games", json=True)
    assert pred["pars"][0]["sents"][0]["words"][0]["lexeme"] == "android mobile"
    assert pred["pars"][0]["sents"][0]["words"][-1]["lexeme"] == "video games"
    assert argmax(pred["cv"]) == clf.get_category_index("science&technology")
    assert [round(p, 5) for p in pred["cv"]] == [0, 0, 0, 0, 0, 0, 4.3789, 0, 0]

    pred = clf.classify("playing football soccer", json=True)
    assert pred["pars"][0]["sents"][0]["words"][-1]["lexeme"] == "football soccer"
    assert argmax(pred["cv"]) == clf.get_category_index("sports")
    assert [round(p, 5) for p in pred["cv"]] == [0, 0, 0, 0, 0, .53463, 0, 1.86708, 0]

    clf.set_model_path("tests/")
    clf.save_model()
    clf.load_model()

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

    rmtree("./tests/tmp", ignore_errors=True)
    rmtree("./tests/ss3_models", ignore_errors=True)


# if __name__ == "__main__":
#     test_pyss3_ss3()
