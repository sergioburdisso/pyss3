<img src="https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/ss3_logo_banner.png" alt="PySS3 Logo" title="PySS3" height="100" />

[![Documentation Status](https://readthedocs.org/projects/pyss3/badge/?version=latest)](http://pyss3.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/sergioburdisso/pyss3.svg?branch=master)](https://travis-ci.org/sergioburdisso/pyss3)
[![Requirements Status](https://requires.io/github/sergioburdisso/pyss3/requirements.svg?branch=master)](https://requires.io/github/sergioburdisso/pyss3/requirements/?branch=master)
[![PyPI version](https://badge.fury.io/py/pyss3.svg)](https://badge.fury.io/py/pyss3)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sergioburdisso/pyss3/master?filepath=examples)
[![MIT License][license-badge]][license]

---

# :sparkles: A python package implementing a novel text classifier with visualization tools for Explainable AI :sparkles:

The SS3 text classifier is a novel supervised machine learning model for text classification. SS3 was originally introduced in Section 3 of the paper _["A text classification framework for simple and effective early depression detection over social media streams"](https://dx.doi.org/10.1016/j.eswa.2019.05.023)_ (preprint available [here](https://arxiv.org/abs/1905.08772)).

**Some virtues of SS3:**

* It has the **ability to visually explain its rationale**.
* Introduces a **domain-independent** classification model that **does not require feature engineering**.
* Naturally supports **incremental (online) learning** and **incremental classification**.
* Well suited for classification over **text streams**.
* Its 3 **hyperparameters** are **easy-to-understand and intuitive** for humans (it is not an "obscure" model).

**Note:** this package also incorporates different variations of the SS3 classifier, such as the one introduced in _"t-SS3: a text classifier with dynamic n-grams for early risk detection over text streams
"_ (recently submitted to Pattern Recognition Letters, preprint available [here](https://arxiv.org/abs/1911.06147)) which allows SS3 to recognize important word n-grams "on the fly".

## What is PySS3?

[PySS3](https://github.com/sergioburdisso/pyss3) is a Python package that allows you to work with SS3 in a very straightforward, interactive and visual way. In addition to the implementation of the SS3 classifier, PySS3 comes with a set of tools to help you developing your machine learning models in a clearer and faster way. These tools let you analyze, monitor and understand your models by allowing you to see what they have actually learned and why. To achieve this, PySS3 provides you with 3  main components: the ``SS3`` class, the ``Live_Test`` class and the ``PySS3 Command Line`` tool, as pointed out below.


### :point_right: The `SS3` class

which implements the classifier using a clear API (very similar to that of `sklearn`'s models):
````python
    from pyss3 import SS3
    clf = SS3()
    ...
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
````

### :point_right: The `Live_Test` class

which allows you to interactively test your model and visually see the reasons behind classification decisions, **with just one line of code**:
```python
    from pyss3.server import Live_Test
    from pyss3 import SS3

    clf = SS3(name="my_model")
    ...
    clf.fit(x_train, y_train)
    Live_Test.run(clf, x_test, y_test) # <- this one! cool uh? :)
```
As shown in the image below, this will open up, locally, an interactive tool in your browser which you can use to (live) test your models with the documents given in `x_test` (or typing in your own!). This will allow you to visualize and understand what your model is actually learning.

![img](https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/ss3_live_test.gif)

For example, we have uploaded two of these live tests online for you to try out: ["Movie Review (Sentiment Analysis)"](http://tworld.io/ss3/live_test_online/#30305) and ["Topic Categorization"](http://tworld.io/ss3/live_test_online/#30303), both were obtained following the [tutorials](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html#tutorials).

### :point_right: And last but not least, the ``PySS3 Command Line`` tool

This is probably the most useful component of PySS3. When you install the package (for instance by using `pip install pyss3`) a new command ``pyss3`` is automatically added to your environment's command line. This command allows you to access to the _PySS3 Command Line_, an interactive command-line query tool. This tool will let you interact with your SS3 models through special commands while assisting you during the whole machine learning pipeline (model selection, training, testing, etc.). Probably one of its most important features is the ability to automatically (and permanently) record the history of every evaluation result of any type (tests, k-fold cross-validations, grid searches, etc.) that you've performed. This will allow you (with a single command) to interactively visualize and analyze your classifier performance in terms of its different hyper-parameters values (and select the best model according to your needs). For instance, let's perform a grid search with a 4-fold cross-validation on the three [hyperparameters](https://pyss3.readthedocs.io/en/latest/user_guide/ss3-classifier.html#hyperparameters), smoothness(`s`), significance(`l`), and sanction(`p`) as follows:

```console
your@user:/your/project/path$ pyss3
(pyss3) >>> load my_model
(pyss3) >>> grid_search path/to/dataset 4-fold -s r(.2,.8,6) -l r(.1,2,6) -p r(.5,2,6)
```
In this illustrative example, `s` will take 6 different values between 0.2 and 0.8, `l` between 0.1 and 2, and `p` between 0.5 and 2. After the grid search finishes, we can use the following command to open up an interactive 3D plot in the browser:
```console
(pyss3) >>> plot evaluations
```
![img](https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/plot_evaluations.gif)

Each point represents an experiment/evaluation performed using that particular combination of values (s, l, and p). Also, these points are painted proportional to how good the performance was using that configuration of the model. Researchers can interactively change the evaluation metrics to be used (accuracy, precision, recall, f1, etc.) and plots will update "on the fly". Additionally, when the cursor is moved over a data point, useful information is shown (including a "compact" representation of the confusion matrix obtained in that experiment). Finally, it is worth mentioning that, before showing the 3D plots, PySS3 creates a single and portable HTML file in your project folder containing the interactive plots. This allows researchers to store, send or upload the plots to another place using this single HTML file (or even provide a link to this file in their own papers, which would be nicer for readers, plus it would increase experimentation transparency). For example, we have uploaded two of these files for you to see: ["Movie Review (Sentiment Analysis)"](https://pyss3.readthedocs.io/en/latest/_static/ss3_model_evaluation[movie_review_3grams].html) and ["Topic Categorization"](https://pyss3.readthedocs.io/en/latest/_static/ss3_model_evaluation[topic_categorization_3grams].html), both evaluation plots were also obtained following the [tutorials](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html#tutorials).


## The PySS3 Workflow :computer:

PySS3 provides two main types of workflow: classic and "command-line". Both workflows are briefly described below.

### Classic

As usual, importing the needed classes and functions from the package, the user writes a python script to train and test the classifiers. In this workflow, user can use the `PySS3 Command Line` tool to perform model selection (though hyperparameter optimization). 

### Command-Line

The whole process is done using only the `PySS3 Command Line` tool. This workflow provides a faster way to perform experimentations since the user doesn't have to write any python script. Plus, this Command Line tool allows the user to actively interact  "on the fly" with the models being developed.


Note: [tutorials](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html#tutorials) are presented in two versions, one for each workflow type, so that the reader can choose the workflow that best suit her/his needs.



## Want to give PySS3 a try? :eyeglasses: :coffee:

Just go to the [Getting Started](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html) page :D

### Installation


Simply use:
```console
pip install pyss3
```
Or, if you already have installed an old version, update it with:
```console
pip install --upgrade pyss3
```

## Want to contribute to this open-source project? :sparkles::octocat::sparkles:

Thanks for your interest in the project, you're ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)!!
Any kind of help is very welcome, from simple ideas, suggestions, recommendations to any type of improvement on the source code of this repo, in other words, Issues and/or Pull Requests are welcome for any level of improvement, from a small typo to a new feature, help us make PySS3 better :+1:.

Remember that you can use the "Edit" button ('pencil' icon) up the top to [edit any file of this repo directly on GitHub](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository).

In case you're planning to create a **new Pull Request**, for committing to this repo, we follow the Chris Beams' "seven rules of a great Git commit message" from ["How to Write a Git Commit Message"](https://chris.beams.io/posts/git-commit/), so make sure your commits follow them as well.


## Further Readings :scroll:


[Full documentation](https://pyss3.readthedocs.io)

[API documentation](https://pyss3.readthedocs.io/en/latest/api/)

[Paper preprint](https://arxiv.org/abs/1912.09322)


[license-badge]: https://img.shields.io/github/license/boyney123/performance-budgets.svg
[license]: LICENSE.txt