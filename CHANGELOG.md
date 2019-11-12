# ChangeLog

All notable changes to PySS3 will be documented here.

## [0.3.3] 2019-11-12

### Fixed
- Setup and tests fixed


## [0.3.2] 2019-11-12

### Added
- Summary operators: now it is possible to use user-defined summary operators, the following static methods were added to the ``SS3`` class: `summary_op_ngrams`, `summary_op_sentences`, and `summary_op_paragraphs`.


## [0.3.1] 2019-11-11

### Added
- update: some docstrings were improved
- update: the README.md / Pypi Description file.

### Fixed
- Python 2 and 3 compatibility problem with scikit-learn (using version 0.20.1 from now on)
- PyPi: setup.py: `long_description_content_type` set to `'text/markdown'`