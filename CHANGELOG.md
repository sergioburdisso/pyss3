# ChangeLog

All notable changes to PySS3 will be documented here.

## [0.6.4] 2021-01-30

### Fixed

- Quick fix of default compatibility with foreign languages ([#15](https://github.com/sergioburdisso/pyss3/issues/15)).


## [0.6.3] 2020-07-17

### Fixed

- Patches issue [#11](https://github.com/sergioburdisso/pyss3/issues/11).

## [0.6.1] 2020-05-26

### Added
- ``Dataset.load_from_files_multilabel()`` can load documents with no
  labels as well ([31251f8](https://github.com/sergioburdisso/pyss3/commit/31251f8)).

- A ``set_testset_from_files_multilabel()`` function was added to the
  ``Live_Test`` class. This function allows loading multilabel
  datasets from disk Live Test server ([0ddbd6a](https://github.com/sergioburdisso/pyss3/commit/0ddbd6a)).

### Fixed

- Fixed a bug in SS3 hyperparameter initialization ([e2e72f9](https://github.com/sergioburdisso/pyss3/commit/e2e72f9)).

## [0.6.0] 2020-05-24

### Added

PySS3 now fully support multi-label classification! :)

- The ``load_from_files_multilabel()`` function was added to the ``Dataset`` class ([7ece7ce](https://github.com/sergioburdisso/pyss3/commit/7ece7ce), resolved [#6](https://github.com/sergioburdisso/pyss3/issues/6))

- The ``Evaluation`` class now supports multi-label classification (resolved [#5](https://github.com/sergioburdisso/pyss3/issues/5))
  - Add multi-label support to ``train()/fit()`` ([4d00476](https://github.com/sergioburdisso/pyss3/commit/4d00476))
  - Add multi-label support to ``Evaluation.test()`` ([0a897dd](https://github.com/sergioburdisso/pyss3/commit/0a897dd))
  - Add multi-label support to ``show_best and get_best()`` ([ef2419b](https://github.com/sergioburdisso/pyss3/commit/ef2419b))
  - Add multi-label support to ``kfold_cross_validation()`` ([aacd3a0](https://github.com/sergioburdisso/pyss3/commit/aacd3a0))
  - Add multi-label support to ``grid_search()`` ([925156d](https://github.com/sergioburdisso/pyss3/commit/925156d), [79f1e9d](https://github.com/sergioburdisso/pyss3/commit/79f1e9d))
  - Add multi-label support to the 3D Evaluation Plot ([42bbc65](https://github.com/sergioburdisso/pyss3/commit/42bbc65))

- The Live Test tool now supports multi-label classification as well ([15657ee](https://github.com/sergioburdisso/pyss3/commit/15657ee), [b617bb7](https://github.com/sergioburdisso/pyss3/commit/b617bb7), resolved [#9](https://github.com/sergioburdisso/pyss3/issues/9))

- Category names are no longer case-insensitive ([4ec009a](https://github.com/sergioburdisso/pyss3/commit/4ec009a), resolved [#8](https://github.com/sergioburdisso/pyss3/issues/8))

## [0.5.7] 2020-05-05

### Added

- The Live Test Tool now supports custom (user-defined) preprosessing methods ([b50cfaf](https://github.com/sergioburdisso/pyss3/commit/b50cfaf), [7c6b0c6](https://github.com/sergioburdisso/pyss3/commit/7c6b0c6), resolved [#3](https://github.com/sergioburdisso/pyss3/issues/3)).

- The tokenization process was improved ([26fff88](https://github.com/sergioburdisso/pyss3/commit/26fff88), [4af8e80](https://github.com/sergioburdisso/pyss3/commit/4af8e80)).

- The process for recognizing word n-grams during classification was improved ([2ceb148](https://github.com/sergioburdisso/pyss3/commit/2ceb148)).

## [0.5.5] 2020-03-02

### Added

- The ``predict`` method was optimized. Now it is 10x to 200x faster! This improvement also has a positive impact on other methods that use ``predict`` such as ``grid_search`` ([37202d8](https://github.com/sergioburdisso/pyss3/commit/37202d8)).
- A new `get_ngrams_length` method was added to ``SS3`` class. It can be used to get the length of longest learned n-gram ([b4f8827](https://github.com/sergioburdisso/pyss3/commit/b4f8827)).
- The Evaluation 3D Plot's GUI was improved ([1bb1e5a](https://github.com/sergioburdisso/pyss3/commit/1bb1e5a)).

### Fixed

- Some bugs and error were fixed ([bc5c4ed](https://github.com/sergioburdisso/pyss3/commit/bc5c4ed), [0d3d7e1](https://github.com/sergioburdisso/pyss3/commit/0d3d7e1), [86a0189](https://github.com/sergioburdisso/pyss3/commit/86a0189), [b0b3eaa](https://github.com/sergioburdisso/pyss3/commit/b0b3eaa), [5dbdc3a](https://github.com/sergioburdisso/pyss3/commit/5dbdc3a))


## [0.5.0] 2020-02-24

### Added
- A new ``Evaluation`` class to ``pyss3.util`` ([8feeef5](https://github.com/sergioburdisso/pyss3/commit/8feeef5a44ccc26e98f967fe470d0d0521d97f96)).
    - Now the user can import the ``Evaluation`` class to perform model evaluation and hyperparameter optimization. This class not only provide methods to evaluate models but also keeps all the advantages previously provided only through the Command Line tool, such as an evaluation cache that automatically keeps track of the evaluation history and the generation of the interactive 3D evaluation plot.
- ``set_name()`` to ``SS3`` ([5b1c355](https://github.com/sergioburdisso/pyss3/commit/5b1c355070ad66884f4360128cbf4f97d9b018b6)).
- ``train()`` to ``SS3`` as a user-friendly alias of ``fit()`` ([74cb540](https://github.com/sergioburdisso/pyss3/commit/74cb54067e10dfeecf0bb52a05d20d2e84b3b34c)).
- Print now supports nested verbosity regions ([78176ab](https://github.com/sergioburdisso/pyss3/commit/78176abb27f2b8a4e7233820ab93265f5c4ee5d5)).

### Fixed

- Compatibility of progress bars with Jupyter Notebooks ([7848b3e](https://github.com/sergioburdisso/pyss3/commit/7848b3e97d42dfb4121ddddbf3fcbae9e9e6736e), [8d163d9](https://github.com/sergioburdisso/pyss3/commit/8d163d9c1b6391fd32c0c5fc0b6d2190376d7f1f), [2029c37](https://github.com/sergioburdisso/pyss3/commit/2029c37af1e7739865402f4af194cd7fc122a2f8), [2a700d5](https://github.com/sergioburdisso/pyss3/commit/2a700d53c5d676c5bbba2cc21494f596d05fbfd2)).
- Bug in SS3.fit when given an empty document ([31eccbc](https://github.com/sergioburdisso/pyss3/commit/31eccbcb193efd3c8ebdacbae12615f54528c37e)).
- Non-string category labels support ([5b1c355](https://github.com/sergioburdisso/pyss3/commit/5b1c355070ad66884f4360128cbf4f97d9b018b6)).
- Issue with verbosity level consistency ([b38d8b0](https://github.com/sergioburdisso/pyss3/commit/b38d8b0bc76c601931da45e8c2c96ff0ad95fda4)).
- IndexError in classify_(multi)label ([fa91952](https://github.com/sergioburdisso/pyss3/commit/fa919523205ac9b49a8761734efc1766b44fe5f5)).
- Python 2 UnicodeEncodeError issue ([867026e](https://github.com/sergioburdisso/pyss3/commit/867026e30ee0566ce02836132ffd4933e18e8e1c)).


## [0.4.1] 2020-02-16

### Added
- Public methods for the SS3's ``cv``, ``gv``, ``lv``, ``sg`` and ``sn`` functions have been added to the ``SS3 class`` ([ef35b25](https://github.com/sergioburdisso/pyss3/commit/ef35b25d8e194569007e6274cbbde856941f5627)). These functions were originally defined in Section 3.2.2 of the [original paper](https://arxiv.org/pdf/1905.08772.pdf).
- Slightly improving training time (due to previously disabled 'by-default' cache of "local value" function).

### Fixed

- A bug on the HTTP Live Test Server ([d106d68](https://github.com/sergioburdisso/pyss3/commit/d106d68bad782c3e5bab9376fc7c4ec52a97cc5c))
- Some bug on the Command-Line tool ([cd42b61](https://github.com/sergioburdisso/pyss3/commit/cd42b61c5c3e163f3aa5e7410fbeb27bb2180363), [8745603](https://github.com/sergioburdisso/pyss3/commit/874560356b439985e676b2a239958f4cb226368a), [dfe8b95](https://github.com/sergioburdisso/pyss3/commit/dfe8b952fadd7082b83f529110dd5e31b0a3e075))

## [0.4.0] 2020-02-11

Among other minor improvements and changes, the most important ones that were added are:

### Added

- ``SS3`` class:
  - The classifier now explicitly supports multi-label classification:
    - Created the following two methods in ``SS3`` class: [``classify_multilabel``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.SS3.classify_multilabel) and [``classify_label``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.SS3.classify_label) ([0759bca](https://github.com/sergioburdisso/pyss3/commit/0759bca4392b2441d8a3668c8aca6bd85791e06f)).
    - A ``multilabel`` argument was added to the ``predict`` method ([c5ac946](https://github.com/sergioburdisso/pyss3/commit/c5ac94681196fb5f7b22fe39a9f6b5bda5362d13)). 
  - A new [``extract_insight()``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.SS3.extract_insight)  method was added to the ``SS3`` class. This method, given a document, returns the pieces of text that were involved in the classification decision ([eee1e29](https://github.com/sergioburdisso/pyss3/commit/eee1e292f410679ea3d25ba45bc1e70c57a3613c)).
  - Created four new methods to allow the user to set the delimiters ([b632fe0](https://github.com/sergioburdisso/pyss3/commit/b632fe05526ed7596b49867094a56718e6fbc219))
    - [``set_block_delimiters``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.SS3.set_block_delimiters)
    - [``set_delimiter_paragraph``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.SS3.set_delimiter_paragraph)
    - [``set_delimiter_sentence``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.SS3.set_delimiter_sentence)
    - [``set_delimiter_word``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.SS3.set_delimiter_word)
- Live Test tool:
  - Improved the the interface by which "Live Test" Server was called from source code, now its usage is more user-friendly and less misleading (read [516b526](https://github.com/sergioburdisso/pyss3/commit/516b52685da3649dfcb64650d3cdaf4ee5ae8d3a) for more info).
  - Improved the way by which multi-label classification was carried out in the Web interface ([046f9f4](https://github.com/sergioburdisso/pyss3/commit/046f9f424a241ce0cdef833d2561ff80bb3f5b2e)).
- Improved how PySS3 handles verbosity levels (read [216be41](https://github.com/sergioburdisso/pyss3/commit/216be41e4824f60071be219ce783134528cde795) for more info ): created the [``set_verbosity()``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.set_verbosity) function.

## [0.3.9] 2019-11-27

### Added
- Live Test: layout updated.
- PySS3 Command Line: ``frange`` function added as an alias of ``r`` for the ``grid_search`` command.

### Fixed
- PySS3 Command Line: live_test always lunch the server with no documents (even when before "live_test a/path")
- Live Test:sentences starting with "unknown" token were not included in the "Advanced" interactive chart


## [0.3.8] 2019-11-25

### Fixed
- Server: fixed bug that stopped the server when receiving arbitrary bytes (not utf-8 strings)
- PySS3 Command Line: fixed bug when loading live_test with a non existing path
- Live Test: now the user can select single letter words (and are also included in the "advanced" live chart)


## [0.3.7] 2019-11-22

### Added
- Summary operators are not longer static.
- ``Server.set_testset_from_files`` lazy load.

### Fixed
- Evaluation plot: confusion matrices size when working with k-folds


## [0.3.6] 2019-11-14

### Added
- `Dataset` class added to `pyss3.util` as an interface to help the user to load/read datasets. Method `Dataset.load_from_files` added
- Documentations updated


## [0.3.5] 2019-11-12

### Added
- PySS3 Command Line Python 2 full compatibility support

### Fixed
- Matplotlib set_yaxis bug fixed


## [0.3.4] 2019-11-12

### Fixed
- Dependencies and compatibility with python 2 Improved


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