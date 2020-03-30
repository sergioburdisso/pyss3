<img src="https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/ss3_logo_banner.png" alt="PySS3 Logo" title="PySS3" height="150" />

[![Documentation Status](https://readthedocs.org/projects/pyss3/badge/?version=latest)](http://pyss3.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/sergioburdisso/pyss3.svg?branch=master)](https://travis-ci.org/sergioburdisso/pyss3)
[![Requirements Status](https://requires.io/github/sergioburdisso/pyss3/requirements.svg?branch=master)](https://requires.io/github/sergioburdisso/pyss3/requirements/?branch=master)
[![PyPI version](https://badge.fury.io/py/pyss3.svg)](https://badge.fury.io/py/pyss3)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e6ffcddd4fe84bbcaefeae09027943b7)](https://www.codacy.com/manual/sergioburdisso/pyss3?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=sergioburdisso/pyss3&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/sergioburdisso/pyss3/branch/master/graph/badge.svg)](https://codecov.io/gh/sergioburdisso/pyss3)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sergioburdisso/pyss3/master?filepath=examples)

---

# :sparkles: A python package implementing a novel text classifier with visualization tools for Explainable AI :sparkles:

The SS3 text classifier is a novel supervised machine learning model for text classification. SS3 was originally introduced in Section 3 of the paper _["A text classification framework for simple and effective early depression detection over social media streams"](https://dx.doi.org/10.1016/j.eswa.2019.05.023)_ (preprint available [here](https://arxiv.org/abs/1905.08772)).

**Some virtues of SS3:**

* It has the **ability to naturally explain its rationale**.
* It is robust to **the class-imbalance problem** since it learns a (special kind of) language model for each class (making the relative difference in the number of documents among classes irrelevant).
* Naturally supports both, **multinomial and multi-label classification**.
* Naturally supports **incremental (online) learning** and **incremental classification**.
* Well suited for classification over **text streams**.
* It is not an "obscure" model since it **only has 3 semantically well-defined hyperparameters** which are easy-to-understand.

**Note:** this package also incorporates different variations of the SS3 classifier, such as the one introduced in _"t-SS3: a text classifier with dynamic n-grams for early risk detection over text streams
"_ (recently submitted to Pattern Recognition Letters, preprint available [here](https://arxiv.org/abs/1911.06147)) which allows SS3 to recognize important word n-grams "on the fly".

## What is PySS3?

[PySS3](https://github.com/sergioburdisso/pyss3) is a Python package that allows you to work with SS3 in a very straightforward, interactive and visual way. In addition to the implementation of the SS3 classifier, PySS3 comes with a set of tools to help you developing your machine learning models in a clearer and faster way. These tools let you analyze, monitor and understand your models by allowing you to see what they have actually learned and why. To achieve this, PySS3 provides you with 3  main components: the ``SS3`` class, the ``Live_Test`` class, and the ``Evaluation`` class, as pointed out below.


### :point_right: The ``SS3`` class

which implements the classifier using a clear API (very similar to that of `sklearn`'s models):
````python
    from pyss3 import SS3
    clf = SS3()
    ...
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
````

Also, this class provides a handful of other useful methods, such as, for instance, [``classify_multilabel``](https://pyss3.rtfd.io/en/latest/api/index.html#pyss3.SS3.classify_multilabel) to provide [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) support: 

````python
    doc = "Liverpool CEO Peter Moore on Building a Global Fanbase"
    
    # standard "single-label" classification
    label = clf.classify_label(doc) # 'business'

    # multi-label classification
    labels = clf.classify_multilabel(doc)  # ['business', 'sports']
````
or [``extract_insight``](https://pyss3.rtfd.io/en/latest/api/index.html#pyss3.SS3.extract_insight) to allow you to [extract the text fragments involved in the classification decision](https://pyss3.readthedocs.io/en/latest/tutorials/extract-insight.html).

### :point_right: The ``Live_Test`` class

which allows you to interactively test your model and visually see the reasons behind classification decisions, **with just one line of code**:
```python
    from pyss3.server import Live_Test
    from pyss3 import SS3

    clf = SS3()
    ...
    clf.fit(x_train, y_train)
    Live_Test.run(clf, x_test, y_test) # <- this one! cool uh? :)
```
As shown in the image below, this will open up, locally, an interactive tool in your browser which you can use to (live) test your models with the documents given in `x_test` (or typing in your own!). This will allow you to visualize and understand what your model is actually learning.

![img](https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/ss3_live_test.gif)

For example, we have uploaded two of these live tests online for you to try out: ["Movie Review (Sentiment Analysis)"](http://tworld.io/ss3/live_test_online/#30305) and ["Topic Categorization"](http://tworld.io/ss3/live_test_online/#30303), both were obtained following the [tutorials](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html#tutorials).

### :point_right: And last but not least, the ``Evaluation`` class

This is probably one of the most useful components of PySS3. As the name may suggest, this class provides the user easy-to-use methods for model evaluation and hyperparameter optimization, like, for example, the [``test``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.util.Evaluation.test), [``kfold_cross_validation``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.util.Evaluation.kfold_cross_validation), [``grid_search``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.util.Evaluation.grid_search), and [``plot``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.util.Evaluation.plot) methods for performing tests, stratified k-fold cross validations, grid searches for hyperparameter optimization, and visualizing evaluation results using an interactive 3D plot, respectively. Probably one of its most important features is the ability to automatically (and permanently) record the history of evaluations that you've performed. This will save you a lot of time and will allow you to interactively visualize and analyze your classifier performance in terms of its different hyper-parameters values (and select the best model according to your needs). For instance, let's perform a grid search with a 4-fold cross-validation on the three [hyperparameters](https://pyss3.rtfd.io/en/latest/user_guide/ss3-classifier.html#hyperparameters), smoothness(`s`), significance(`l`), and sanction(`p`):

```python
from pyss3.util import Evaluation
...
best_s, best_l, best_p, _ = Evaluation.grid_search(
    clf, x_train, y_train,
    s=[0.2 , 0.32, 0.44, 0.56, 0.68, 0.8],
    l=[0.1 , 0.48, 0.86, 1.24, 1.62, 2],
    p=[0.5, 0.8, 1.1, 1.4, 1.7, 2],
    k_fold=4
)
```
In this illustrative example, `s`, `l`, and `p` will take those 6 different values each, and once the search is over, this function will return (by default) the hyperparameter values that obtained the best accuracy.
Now, we could also use the ``plot`` function to analyze the results obtained in our grid search using the interactive 3D evaluation plot:

```python
Evaluation.plot()
```

![img](https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/plot_evaluations.gif)

In this 3D plot, each point represents an experiment/evaluation performed using that particular combination of values (s, l, and p). Also, these points are painted proportional to how good the performance was according to the selected metric; the plot will update "on the fly" when the user select a different evaluation metric (accuracy, precision, recall, f1, etc.). Additionally, when the cursor is moved over a data point, useful information is shown (including a "compact" representation of the confusion matrix obtained in that experiment). Finally, it is worth mentioning that, before showing the 3D plots, PySS3 creates a single and portable HTML file in your project folder containing the interactive plots. This allows users to store, send or upload the plots to another place using this single HTML file. For example, we have uploaded two of these files for you to see: ["Sentiment Analysis (Movie Reviews)"](https://pyss3.readthedocs.io/en/latest/_static/ss3_model_evaluation[movie_review_3grams].html) and ["Topic Categorization"](https://pyss3.readthedocs.io/en/latest/_static/ss3_model_evaluation[topic_categorization_3grams].html), both evaluation plots were also obtained following the [tutorials](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html#tutorials).


## The PySS3 Workflow :computer:

PySS3 provides two main types of workflow: classic and "command-line". Both workflows are briefly described below.

### Classic

As usual, importing the needed classes and functions from the package, the user writes a python script to train and test the classifiers.

### Command-Line

When you install the package (for instance by using `pip install pyss3`) a new command ``pyss3`` is automatically added to your environment's command line. This command allows you to access to the _PySS3 Command Line_, an interactive command-line query tool. This workflow consist of using this tool to carry out the whole machine learning pipeline (model selection, training, testing, etc.), which provides a faster way to perform experimentations since the user doesn't have to write any python script. Plus, this Command Line tool allows the user to actively interact  "on the fly" with the models being developed.


Note: [tutorials](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html#tutorials) are presented in two versions, one for each workflow type, so that the reader can choose the workflow that best suit her/his needs.



## Want to give PySS3 a shot? :eyeglasses: :coffee:

Just go to the [Getting Started](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html) page :D

### Installation


Simply use:
```console
pip install pyss3
```
Or, if you already have installed an old version, update it with:
```console
pip install -U pyss3
```

## Want to contribute to this Open Source project? :sparkles::octocat::sparkles:

Thanks for your interest in the project, you're ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)!!
Any kind of help is very welcome (Code, Bug reports, Content, Data, Documentation, Design, Examples, Ideas, Feedback, etc.),  Issues and/or Pull Requests are welcome for any level of improvement, from a small typo to new features, help us make PySS3 better :+1:

Remember that you can use the "Edit" button ('pencil' icon) up the top to [edit any file of this repo directly on GitHub](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository).

Also, if you star this repo (:star2:), you would be helping PySS3 to gain more visibility and reach the hands of people who may find it useful since repository lists and search results are usually ordered by the total number of stars.

Finally, in case you're planning to create a **new Pull Request**, for committing to this repo, we follow the "seven rules of a great Git commit message" from ["How to Write a Git Commit Message"](https://chris.beams.io/posts/git-commit/), so make sure your commits follow them as well.

(please do not hesitate to send me an email to sergio.burdisso@gmail.com for anything)

<!-- ### Contributors :blue_heart:

Thanks goes to these awesome people ([emoji key](https://allcontributors.org/docs/en/emoji-key -->

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->


## Further Readings :scroll:


[Full documentation](https://pyss3.readthedocs.io)

[API documentation](https://pyss3.readthedocs.io/en/latest/api/)

[Paper preprint](https://arxiv.org/abs/1912.09322)
