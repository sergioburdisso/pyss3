# -*- coding: utf-8 -*-
"""Tests for pyss3.cmd_line."""
from pyss3.cmd_line import SS3Prompt, main
from pyss3.server import Server
from pyss3.util import Evaluation
from pyss3 import SS3
from os import path

import pyss3.cmd_line
import pyss3.util
import pytest
import sys

MODEL_NAME = "cmd_test"
DATASET_FOLDER = "dataset"
DATASET_FOLDER_MR = "dataset_mr"
ArgsParserError = "1 2 3 4"
PYTHON3 = sys.version_info[0] >= 3
dataset_path = path.join(path.abspath(path.dirname(__file__)), DATASET_FOLDER)
dataset_path_mr = path.join(path.abspath(path.dirname(__file__)), DATASET_FOLDER_MR)


def test_ss3prompt(mocker, monkeypatch):
    """Test the Command-Line."""
    if PYTHON3:
        monkeypatch.setattr('builtins.input', lambda: 'Y')
    mocker.patch.object(SS3Prompt, "cmdloop")

    # not working in Python 2
    # mocker.patch("pyss3.cmd_line.STOPWORDS_FILE", "tests/ss3_models/ss3_stopwords[%s].txt")
    # mocker.patch(
    #     "pyss3.util.EVAL_HTML_OUT_FILE",
    #     "tests/ss3_models/ss3_model_evaluation[%s].html"
    # )
    # replaced by:
    html_file_original = pyss3.util.EVAL_HTML_OUT_FILE
    pyss3.cmd_line.STOPWORDS_FILE = "tests/ss3_models/ss3_stopwords[%s].txt"
    pyss3.util.EVAL_HTML_OUT_FILE = "tests/ss3_models/" + html_file_original

    Evaluation.__cache__ = None
    Evaluation.__cache_file__ = None
    Evaluation.__clf__ = None
    Evaluation.__last_eval_tag__ = None
    Evaluation.__last_eval_method__ = None
    Evaluation.__last_eval_def_cat__ = None

    main()

    SS3.__models_folder__ = "tests/ss3_models"

    cmd = SS3Prompt()

    cmd.do_train(dataset_path + " file")

    # do_new
    cmd.do_new(MODEL_NAME + "_train_folder")
    cmd.do_train(dataset_path_mr + " 3-grams")
    cmd.do_train("non-existing 3-grams")
    cmd.do_train(dataset_path_mr)  # no test documents
    cmd.do_train(ArgsParserError)

    # do_new
    cmd.do_new(MODEL_NAME)

    # do_save evaluations - empty result history case
    cmd.do_save("evaluations")

    # do_train
    cmd.do_train(dataset_path + " 3-grams file")
    cmd.do_train("non-existing 3-grams file")
    cmd.do_train(dataset_path)  # no test documents
    cmd.do_train(ArgsParserError)

    # do_next_word
    cmd.do_next_word("android")
    cmd.do_next_word('')  # doesn't work ¯\_(ツ)_/¯

    if PYTHON3:
        mocker.patch("matplotlib.pyplot.show")
        # do_test
        cmd.do_test(dataset_path + " file")
        cmd.do_test(dataset_path + " file")  # cache
        cmd.do_test(dataset_path + " file unknown")  # def_cat=unknown
        cmd.do_test(dataset_path + " file xxx")  # def_cat=xxx
        cmd.do_test(dataset_path + " file s .5 p")  # IndexError
        cmd.do_test(dataset_path + " file s .5 l 'a'")  # BaseException
        cmd.do_test("not-a-directory")
        cmd.do_test(dataset_path)  # no documents
        cmd.do_test(dataset_path_mr + " file")  # no documents
        cmd.do_test(dataset_path_mr)  # unknown categories
        cmd.do_test(ArgsParserError)

        # do_k_fold
        cmd.do_k_fold(dataset_path + " file 3-grams 3-fold")
        cmd.do_k_fold(dataset_path + " file 3-grams 3-fold")  # cache
        cmd.do_k_fold(dataset_path + " file 3-grams 3-fold xxx")
        cmd.do_k_fold(dataset_path_mr + " 3-grams 3-fold xxx")
        cmd.do_k_fold(ArgsParserError)

    # do_grid_search
    cmd.do_grid_search(dataset_path + " file 3-fold unknown p [.2] l [.2] s [.2]")
    cmd.do_grid_search(dataset_path + " file 2-grams unknown p [.2] l [.2] s [.2]")
    cmd.do_grid_search(dataset_path + " file 2-grams xxx p [.2] l [.2] s [.2]")
    cmd.do_grid_search(dataset_path_mr + " 2-grams p [.2] l [.2] s [.2]")
    cmd.do_grid_search(ArgsParserError)

    # do_evaluations
    mocker.patch("webbrowser.open")
    cmd.do_evaluations("info")
    cmd.do_evaluations("save")
    cmd.do_evaluations("plot")

    # do_load
    cmd.do_load(MODEL_NAME)
    if PYTHON3:
        cmd.do_evaluations("remove p " + str(SS3.__p__))
        cmd.do_evaluations("remove test s .2 l .2 p .2")
        cmd.do_evaluations("remove test")
        cmd.do_evaluations("remove 3-fold")

    cmd.do_evaluations("otherwise")
    cmd.do_evaluations(ArgsParserError)

    # do_classify
    cmd.do_classify(dataset_path + "/food.txt")
    monkeypatch.setattr('sys.stdin.readlines', lambda: 'nice food!\n')
    cmd.do_classify("")
    cmd.do_classify(ArgsParserError)

    # do_live_test
    mocker.patch.object(Server, "set_testset")
    mocker.patch.object(Server, "serve")
    cmd.do_live_test("path")
    set_testset_from_files = mocker.patch.object(Server, "set_testset_from_files")
    set_testset_from_files.return_value = True
    cmd.do_live_test("path")
    set_testset_from_files.return_value = False
    cmd.do_live_test("path")
    cmd.do_live_test("")
    cmd.do_live_test(ArgsParserError)

    # do_learn
    cmd.do_learn(ArgsParserError)
    cmd.do_learn("food 3-grams %s/food.txt" % dataset_path)

    # do_update
    cmd.do_update('')

    # do_info
    cmd.do_info("all")
    cmd.do_info("evaluations")

    # do_debug_term
    cmd.do_debug_term("android")
    cmd.do_debug_term('')  # doesn't work ¯\_(ツ)_/¯

    # do_plot
    cmd.do_plot("evaluations")
    if PYTHON3:
        cmd.do_plot("distribution food")

    cmd.do_plot("distribution")
    cmd.do_plot("distribution non-existing")
    cmd.do_plot(ArgsParserError)

    # do_set
    cmd.do_set("s .5")
    cmd.do_set(ArgsParserError)

    # do_get
    cmd.do_get("s")
    cmd.do_get("l")
    cmd.do_get("p")
    cmd.do_get("a")
    cmd.do_get("otherwise")
    cmd.do_get(ArgsParserError)

    # do_save
    cmd.do_save("model")
    cmd.do_save("evaluations")
    mocker.patch.object(SS3, "save_cat_vocab")
    mocker.patch.object(SS3, "save_vocab")
    cmd.do_save("vocabulary")
    cmd.do_save("vocabulary food")
    cmd.do_save("vocabulary invalid-category")
    cmd.do_save("stopwords")
    cmd.do_save("stopwords .01")
    mocker.patch.object(SS3, "get_stopwords").return_value = []
    cmd.do_save("stopwords")
    cmd.do_save(ArgsParserError)

    # do_load
    cmd.do_load(MODEL_NAME)

    # do_clone
    cmd.do_clone(MODEL_NAME + "_backup")
    cmd.do_clone(MODEL_NAME + "_copy")

    # do_rename
    cmd.do_rename(MODEL_NAME + "_renamed")
    if PYTHON3:
        cmd.do_rename(MODEL_NAME)
        mocker.patch("pyss3.cmd_line.MODELS", [MODEL_NAME])
        cmd.do_new(MODEL_NAME)
        cmd.do_load(MODEL_NAME + "_copy")
        cmd.do_rename(MODEL_NAME)

    # do_license
    cmd.do_license("")

    # do_exit
    with pytest.raises(SystemExit):
        cmd.do_exit("")

    cmd.complete_info("", 0, 0, 0)
    cmd.complete_save("", 0, 0, 0)
    cmd.complete_load("", 0, 0, 0)
    cmd.complete_train("", 0, 0, 0)
    cmd.complete_test("", 0, 0, 0)
    cmd.complete_live_test("", 0, 0, 0)
    cmd.complete_learn("", 0, 0, 0)
    cmd.complete_set("", 0, 0, 0)
    cmd.complete_plot("", 0, 0, 0)
    cmd.complete_grid_search("", 0, 0, 0)
    cmd.complete_evaluations("", 0, 0, 0)

    main()

    pyss3.util.EVAL_HTML_OUT_FILE = html_file_original
