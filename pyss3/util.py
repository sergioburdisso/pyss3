# -*- coding: utf-8 -*-
"""This is a helper module with utility classes and functions."""
from __future__ import print_function
from io import open
from os import listdir, path
from tqdm import tqdm

import unicodedata
import re

ENCODING = "utf-8"

re_url_noise = "(?P<url_noise>%s|%s|%s|%s)" % (
    r"(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}"
    r"\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
    r"(?:\.[a-zA-Z]{2,5}\?[^\s]+ )",
    r"(?:/[a-zA-Z0-9_]+==)",
    r"(?:(([a-zA-Z]+://)?([a-zA-Z]+\.)?([a-zA-Z]{1,7}\.[a-zA-Z]{2,7})?"
    r"/[a-zA-Z0-9]*[A-Z0-9][a-zA-Z0-9]*(\?[a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*)?)|"
    r"((\?[a-zA-Z0-9]+=[a-zA-Z0-9./:=?#_-]*)"
    r"(&[a-zA-Z0-9]+=[a-zA-Z0-9./:=?#_-]*)*) )"
)
regex_remove_url_noise = re.compile(re_url_noise)
regex_camelCase = re.compile(r"#(?P<camel>[A-Z][a-z]+|[A-Z][A-Z]+)")
regex_date = re.compile(
    r"(?:\d+([.,]\d+)?[-/\\]\d+([.,]\d+)?[-/\\]\d+([.,]\d+)?)"
)
regex_temp = re.compile(
    r"(?:\d+([.,]\d+)?\s*(\xc2[\xb0\xba])?\s*[CcFf](?=[^a-zA-Z]))"
)
regex_money = re.compile(r"(?:\$\s*\d+([.,]\d+)?)")
regex_percent = re.compile(r"(?:\d+([.,]\d+)?\s*%)")
regex_number = re.compile(r"(?:\d+([.,]\d+)?)")
regex_dots_chars = re.compile(r"(?:([(),;:?!=\"/.|<>\[\]]+)|(#(?![a-zA-Z])))")
regex_dots_chained = re.compile(r"(?:(#[a-zA-Z0-9]+)(\s)*(?=#))")


class VERBOSITY:
    """verbosity "enum" constants."""

    QUIET = 0
    NORMAL = 1
    VERBOSE = 2


class Dataset:
    """A helper class with methods to read/write datasets."""

    @staticmethod
    def load_from_files(data_path, folder_label=True, as_single_doc=False):
        """
        Load category documents from disk.

        :param data_path: the training or the test set path
        :type data_path: str
        :param folder_label: if True, read category labels from folders,
                             otherwise, read category labels from file names.
                             (default: True)
        :type folder_label: bool
        :param as_single_doc: read the documents as a single (and big) document
                              (default: False)
        :type folder_label: bool
        :returns: the (x_train, y_train) or the (x_test, y_test) pairs.
        :rtype: tuple
        """
        x_data = []
        y_data = []
        cat_info = {}

        Print.info("reading files...")

        if not folder_label:

            files = listdir(data_path)
            for file in tqdm(files, desc=" Category files",
                             leave=False, disable=Print.is_quiet()):
                file_path = path.join(data_path, file)
                if path.isfile(file_path):
                    cat = path.splitext(file)[0]

                    with open(file_path, "r", encoding=ENCODING) as fcat:
                        docs = (fcat.readlines()
                                if not as_single_doc else [fcat.read()])

                    x_data.extend(docs)
                    y_data.extend([cat] * len(docs))

                    cat_info[cat] = len(docs)
        else:

            folders = listdir(data_path)
            for item in tqdm(folders, desc=" Categories",
                             leave=False, disable=Print.is_quiet()):
                item_path = path.join(data_path, item)
                if not path.isfile(item_path):
                    cat_info[item] = 0
                    files = listdir(item_path)
                    for file in tqdm(files, desc=" Documents",
                                     leave=False, disable=Print.is_quiet()):
                        file_path = path.join(item_path, file)
                        if path.isfile(file_path):
                            with open(file_path, "r", encoding=ENCODING) as ffile:
                                x_data.append(ffile.read())
                                y_data.append(item)
                            cat_info[item] += 1

        Print.info("%d categories found" % len(cat_info))
        for cat in cat_info:
            Print.info(
                "'%s'%s"
                %
                (
                    cat,
                    '' if as_single_doc else " (%d documents)" % cat_info[cat]
                ),
                offset=4
            )

        return x_data, y_data

    # TODO: save_to_files(x_train, y_train, x_test, y_test)


class RecursiveDefaultDict(dict):
    """A dict whose default value is a dict."""

    def __missing__(self, key):
        """Class constructor."""
        value = self[key] = type(self)()
        return value


class Preproc:
    """A helper class with methods to preprocess input documents."""

    @staticmethod
    def clean_and_ready(text, dots=True, normalize=True, min_len=1):
        """Clean and prepare the text."""
        if normalize:
            try:
                text = text.decode("utf-8")
            except BaseException:
                pass

            text = ''.join(
                c for c in unicodedata.normalize('NFKD', text)
                if unicodedata.category(c) != 'Mn'
            ).replace("'", '')

        # removing tweets links, noisy parameters in urls, etc.
        text = regex_remove_url_noise.sub(".", text)

        # resolving camel-cased words (e.g. #ThisTypeOfHashTags)
        text = regex_camelCase.sub(r" \1 ", text)

        # tokenizing terms related to numbers
        text = regex_number.sub(
            "NNBRR",
            regex_percent.sub(
                "NNBRRP",
                regex_money.sub(
                    "MNNBRR",
                    regex_date.sub(
                        "NNBRRD",
                        regex_temp.sub(
                            "NNBRRT",
                            text + " "
                        )
                    )
                )
            )
        )

        # cleaning up the text
        if not dots:
            return (
                " ".join(re.findall("[a-zA-Z\n]{%d,}" % min_len, text))
            ).lower()
        else:
            text = regex_dots_chars.sub(
                ".",
                regex_dots_chained.sub(r"\1.", text)
            ) + "."
            return (
                " ".join(re.findall(r"[a-zA-Z\n]{%d,}|\." % min_len, text))
            ).lower()


class Style:
    """Helper class to handle print styles."""

    __header__ = '\033[95m'
    __okblue__ = '\033[94m'
    __okgreen__ = '\033[92m'
    __warning__ = '\033[93m'
    __fail__ = '\033[91m'
    __endc__ = '\033[0m'
    __bold__ = '\033[1m'
    __underline__ = '\033[4m'

    @staticmethod
    def __apply__(text, format_str):
        """Appply a given style."""
        return "%s%s%s" % (format_str, text, Style.__endc__)

    @staticmethod
    def bold(text):
        """Apply bold style to ``text``."""
        return Style.__apply__(text, Style.__bold__)

    @staticmethod
    def ubold(text):
        """Apply underline and bold style to ``text``."""
        return Style.underline((Style.bold(text)))

    @staticmethod
    def underline(text):
        """Underline ``text``."""
        return Style.__apply__(text, Style.__underline__)

    @staticmethod
    def fail(text):
        """Apply the 'fail' style to ``text``."""
        return Style.__apply__(text, Style.__fail__)

    @staticmethod
    def warning(text):
        """Apply the 'warning' style to ``text``."""
        return Style.__apply__(text, Style.__warning__)

    @staticmethod
    def green(text):
        """Apply 'green' style to ``text``."""
        return Style.__apply__(text, Style.__okgreen__)

    @staticmethod
    def blue(text):
        """Apply 'blue' style to ``text``."""
        return Style.__apply__(text, Style.__okblue__)

    @staticmethod
    def header(text):
        """Apply 'header' style to ``text``."""
        return Style.__apply__(text, Style.__header__)


class Print:
    """Helper class to handle print functionalities."""

    __error_start__ = '*** '
    __error_end__ = ''
    __warn_start__ = '* '
    __warn_end__ = ''
    __info_start__ = '[ '
    __info_end__ = ' ]'

    style = Style

    __verbosity__ = VERBOSITY.NORMAL
    __verbosity_old__ = None

    @staticmethod
    def error(msg, raises=None, offset=0, decorator=True):
        """
        Print an error.

        :param msg: the message to show
        :type msg: str
        :param raises: the exception to be raised after showing the message
        :type raises: Exception
        :param offset: shift the message to the right (``offset`` characters)
        :type offset: int
        :param decorator: if True, use error message decoretor
        :type decorator: bool
        """
        print(
            Style.fail("%s%s%s%s" % (
                " " * offset,
                Print.__error_start__ if decorator else '',
                str(msg),
                Print.__error_end__ if decorator else ''
            ))
        )
        if raises:
            raise raises

    @staticmethod
    def warn(msg, newln=True, raises=None, offset=0, decorator=True):
        """
        Print a warning message.

        :param msg: the message to show
        :type msg: str
        :param newln: use new line after the message (default: True)
        :type newln: bool
        :param raises: the exception to be raised after showing the message
        :type raises: Exception
        :param offset: shift the message to the right (``offset`` characters)
        :type offset: int
        :param decorator: if True, use warning message decoretor
        :type decorator: bool
        """
        Print.show(
            Style.warning("%s%s%s%s" % (
                " " * offset,
                Print.__warn_start__ if decorator else '',
                str(msg),
                Print.__warn_end__ if decorator else '',
            )), newln
        )
        if raises:
            raise raises

    @staticmethod
    def info(msg, newln=True, offset=0, decorator=True, force_show=False):
        """
        Print an info message.

        :param msg: the message to show
        :type msg: str
        :param newln: use new line after the message (default: True)
        :type newln: bool
        :param offset: shift the message to the right (``offset`` characters)
        :type offset: int
        :param decorator: if True, use info message decoretor
        :type decorator: bool
        :param force_show: if True, show message even when not in verbose mode
        :type force_show: bool
        """
        if Print.is_verbose() or force_show:
            Print.show(
                Style.blue("%s%s%s%s" % (
                    " " * offset,
                    Print.__info_start__ if decorator else '',
                    str(msg),
                    Print.__info_end__ if decorator else '',
                )), newln
            )

    @staticmethod
    def show(msg='', newln=True, offset=0):
        """
        Print a message.

        :param msg: the message to show
        :type msg: str
        :param newln: use new line after the message (default: True)
        :type newln: bool
        :param offset: shift the message to the right (``offset`` characters)
        :type offset: int
        """
        if not Print.is_quiet():
            print((" " * offset) + str(msg), end='\n' if newln else '')

    @staticmethod
    def set_decorator_info(start, end=None):
        """
        Set info messages decorator.

        :param start: messages preffix
        :type start: str
        :param end: messages suffix
        :type end: str
        """
        Print.__info_start__ = start or Print.__info_start__
        Print.__info_end__ = end or Print.__info_end__

    @staticmethod
    def set_decorator_warn(start, end=None):
        """
        Set warning messages decorator.

        :param start: messages preffix
        :type start: str
        :param end: messages suffix
        :type end: str
        """
        Print.__warn_start__ = start or Print.__warn_start__
        Print.__warn_end__ = end or Print.__warn_end__

    @staticmethod
    def set_decorator_error(start, end=None):
        """
        Set error messages decorator.

        :param start: messages preffix
        :type start: str
        :param end: messages suffix
        :type end: str
        """
        Print.__error_start__ = start or Print.__error_start__
        Print.__error_end__ = end or Print.__error_end__

    @staticmethod
    def set_verbosity(level):
        """
        Set the verbosity level.

            - ``0`` (quiet): do not output any message (only error messages)
            - ``1`` (normal): default behavior, display only warning messages and progress bars
            - ``2`` (verbose): display also the informative non-essential messages

        The following built-in constants can also be used to refer to these 3 values:
        ``VERBOSITY.QUIET``, ``VERBOSITY.NORMAL``, and ``VERBOSITY.VERBOSE``, respectively.

        For example, if you want PySS3 to hide everything, even progress bars, you could do:

        >>> from pyss3.util import Print, VERBOSITY
        ...
        >>> Print.set_verbosity(VERBOSITY.QUIET)  # or, equivalently, Print.set_verbosity(0)
        ...
        >>> # here's the rest of your code :D

        :param level: the verbosity level
        :type level: int
        """
        Print.__verbosity__ = level

    @staticmethod
    def is_quiet():
        """Check if the current verbosity level is quiet."""
        return Print.__verbosity__ == VERBOSITY.QUIET

    @staticmethod
    def is_verbose():
        """Check if the current verbosity level is verbose."""
        return Print.__verbosity__ >= VERBOSITY.VERBOSE

    @staticmethod
    def verbosity_region_begin(level):
        """
        Indicate that a region with different verbosity begins.

        When the region ends by calling ``verbosity_region_end``, the previous
        verbosity will be restored.

        Example:

        >>> from pyss3.util import Print,VERBOSITY
        ...
        >>> Print.verbosity_region_begin(VERBOSITY.QUIET)
        >>> # inside this region (from now on), verbosity will be 'quiet'
        ...
        >>> Print.verbosity_region_end()
        >>> # the verbosity level is restored to what it was before entering the region

        :param level: the verbosity level for this region
                      (see ``set_verbosity`` documentation for valid values)
        :type level: int
        """
        if not Print.is_quiet():
            Print.__verbosity_old__ = Print.__verbosity__
            Print.__verbosity__ = level

    @staticmethod
    def verbosity_region_end():
        """
        Indicate that a region with different verbosity ends.

        The verbosity will be restored to the value it had
        before beginning this region with ``verbosity_region_begin``.

        Example:

        >>> from pyss3.util import Print,VERBOSITY
        ...
        >>> Print.verbosity_region_begin(VERBOSITY.VERBOSE)
        >>> # inside this region (from now on), verbosity will be 'verbose'
        ...
        >>> Print.verbosity_region_end()
        >>> # the verbosity level is restored to what it was before entering the region
        """
        if Print.__verbosity_old__ is not None:
            Print.__verbosity__ = Print.__verbosity_old__
            Print.__verbosity_old__ = None
