# PySS3: A python package implementing a novel text classifier with visualization tools for Explainable AI
[![Documentation Status](https://readthedocs.org/projects/pyss3/badge/?version=latest)](http://pyss3.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/sergioburdisso/pyss3.svg?branch=master)](https://travis-ci.org/sergioburdisso/pyss3)

The SS3 text classifier was originally introduced in Section 3 of the [paper](https://dx.doi.org/10.1016/j.eswa.2019.05.023) entitled _"A text classification framework for simple and effective early depression detection over social media streams"_ (preprint available [here](https://arxiv.org/abs/1905.08772)).

**SS3 highlights:**

* A novel text classifier having the ability to visually explain its rationale.
* Domain-independent classification that does not require feature engineering.
* Naturally supports incremental (online) learning and incremental classification.


## Installation


### PyPi installation

Simply type:

    pip install pyss3


### Installation from source

To install latest version from github, clone the source from the project repository and install with setup.py::

    git clone https://github.com/sergioburdisso/pyss3
    cd pyss3
    python setup.py install
 

## API Documentation


Full API documentation can be found [here](https://pyss3.readthedocs.io)
