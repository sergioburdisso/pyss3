# -*- coding: utf-8 -*-
"""Tests for pyss3.util."""
from os import path
from pyss3.util import Dataset, RecursiveDefaultDict, Print, VERBOSITY

import pytest

DATASET_FOLDER = "dataset_mr"

dataset_path = path.join(path.abspath(path.dirname(__file__)), DATASET_FOLDER)


def test_util():
    """Test utility module."""
    x_train, y_train = Dataset.load_from_files(dataset_path, folder_label=True)

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
