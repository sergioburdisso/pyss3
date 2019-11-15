.. _changelog:

*************
Change Log
*************

All notable changes to PySS3 will be documented here.

[0.3.6] 2019-11-14
==================

Added
-----
- ``Dataset`` class added to ``pyss3.util`` as an interface to help the user to load/read datasets. Method ``Dataset.load_from_files`` added
- Documentations updated

[0.3.5] 2019-11-12
==================

Added
-----
- PySS3 Command Line Python 2 full compatibility support

Fixed
-----
- Matplotlib set_yaxis bug fixed


[0.3.4] 2019-11-12
==================

Fixed
-----
- Dependencies and compatibility with python 2 Improved


[0.3.3] 2019-11-12
==================

Fixed
-----
- Setup and tests fixed


[0.3.2] 2019-11-12
==================

Added
-----
- Summary operators: now it is possible to use user-defined summary operators, the following static methods were added to the ``SS3`` class: `summary_op_ngrams`, `summary_op_sentences`, and `summary_op_paragraphs`.


[0.3.1] 2019-11-11
==================

Added
-----
- update: some docstrings were improved
- update: the README.md / Pypi Description file.

Fixed
-----
- Python 2 and 3 compatibility problem with scikit-learn (using version 0.20.1 from now on)
- PyPi: setup.py: `long_description_content_type` set to `'text/markdown'`