<img src="https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/ss3_logo_banner.png" alt="PySS3 Logo" title="PySS3" height="150" />

[![Documentation Status](https://readthedocs.org/projects/pyss3/badge/?version=latest)](http://pyss3.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://api.travis-ci.com/sergioburdisso/pyss3.svg?branch=master)](https://app.travis-ci.com/github/sergioburdisso/pyss3)
[![codecov](https://codecov.io/gh/sergioburdisso/pyss3/branch/master/graph/badge.svg)](https://codecov.io/gh/sergioburdisso/pyss3)
[![PyPI version](https://badge.fury.io/py/pyss3.svg)](https://badge.fury.io/py/pyss3)
[![Downloads](https://static.pepy.tech/badge/pyss3)](https://pepy.tech/project/pyss3)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sergioburdisso/pyss3/master?filepath=examples)

---

# A Python package implementing a new simple and interpretable model for text classification

:sushi: **Online live demos:** http://tworld.io/ss3/ :icecream::ice_cream::cake:

<br>

The SS3 text classifier is a novel and simple supervised machine learning model for text classification which is interpretable, that is, it has the **ability to naturally (self)explain its rationale**. It was originally introduced in Section 3 of the paper _["A text classification framework for simple and effective early depression detection over social media streams"](https://dx.doi.org/10.1016/j.eswa.2019.05.023)_ ([arXiv preprint](https://arxiv.org/abs/1905.08772)).
This simple model obtained the best and 2nd-best results, consecutively, in the last three editions of the [CLEF's eRisk lab](https://erisk.irlab.org/) among all participating models [[Burdisso *et al.* 2019](http://ceur-ws.org/Vol-2380/paper_103.pdf); [Loyola *et al.* 2021](http://ceur-ws.org/Vol-2936/paper-81.pdf)].
Given its white-box nature, it allows researchers and practitioners to deploy interpretable (i.e. self-explainable) and therefore more reliable, models for text classification (which could be especially useful for those working with classification problems by which people's lives could be somehow affected).

**Note:** this package also incorporates different variations of the original model, such as the one introduced in _["t-SS3: a text classifier with dynamic n-grams for early risk detection over text streams"](https://doi.org/10.1016/j.patrec.2020.07.001)_ ([arXiv preprint](https://arxiv.org/abs/1911.06147)) which allows SS3 to recognize important variable-length word n-grams "on the fly".

## What is PySS3?

[PySS3](https://github.com/sergioburdisso/pyss3) is a Python package that allows you to work with SS3 in a very straightforward, interactive and visual way. In addition to the implementation of the SS3 classifier, PySS3 comes with a set of tools to help you developing your machine learning models in a clearer and faster way. These tools let you analyze, monitor and understand your models by allowing you to see what they have actually learned and why. To achieve this, PySS3 provides you with 3  main components: the ``SS3`` class, the ``Live_Test`` class, and the ``Evaluation`` class, as pointed out below.


### :point_right: The ``SS3`` class

which implements the classifier using a clear API. For instance, let's first load [one of the tutorial](https://pyss3.rtfd.io/en/latest/tutorials/movie-review.html)'s dataset:

```python
from pyss3.util import Dataset

url = "https://github.com/sergioburdisso/pyss3/raw/master/examples/datasets/movie_review.zip"

x_train, y_train = Dataset.load_from_url(url, "train")
x_test, y_test = Dataset.load_from_url(url, "test")
```

Now let's train our first SS3 model! note that the API is very similar to that of `sklearn`'s models:

````python
from pyss3 import SS3

clf = SS3()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
````

Also, this class provides a handful of other useful methods, such as, for instance, [``extract_insight()``](https://pyss3.rtfd.io/en/latest/api/index.html#pyss3.SS3.extract_insight) to [extract the text fragments involved in the classification decision](https://pyss3.readthedocs.io/en/latest/tutorials/extract-insight.html) (allowing you to better understand the rationale behind the model’s predictions) or [``classify_multilabel()``](https://pyss3.rtfd.io/en/latest/api/index.html#pyss3.SS3.classify_multilabel) to provide [multi-label classification](https://en.wikipedia.org/wiki/Multi-label_classification) support: 

````python
doc = "Liverpool CEO Peter Moore on Building a Global Fanbase"

# standard "single-label" classification
label = clf.classify_label(doc) # 'business'

# multi-label classification
labels = clf.classify_multilabel(doc)  # ['business', 'sports']
````

### :point_right: The ``Live_Test`` class

which allows you to interactively test your model and visually see the reasons behind classification decisions, **with just one line of code**:
```python
from pyss3.server import Live_Test

clf = SS3()
clf.fit(x_train, y_train)

Live_Test.run(clf, x_test, y_test) # <- this one! cool uh? :)
```
As shown in the image below, this will open up, locally, an interactive tool in your browser which you can use to (live) test your models with the documents given in `x_test` (or typing in your own!). This will allow you to visualize and understand what your model is actually learning.

![img](https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/ss3_live_test.gif)

For example, we have uploaded two of these live tests online for you to try out: ["Movie Review (Sentiment Analysis)"](http://tworld.io/ss3/live_test_online/#30305) and ["Topic Categorization"](http://tworld.io/ss3/live_test_online/#30303), both were obtained following the [tutorials](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html#tutorials).

### :point_right: And last but not least, the ``Evaluation`` class

This is probably one of the most useful components of PySS3. As the name suggests, this class provides the user easy-to-use methods for model evaluation and hyperparameter optimization, like, for example, the [``test``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.util.Evaluation.test), [``kfold_cross_validation``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.util.Evaluation.kfold_cross_validation), [``grid_search``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.util.Evaluation.grid_search), and [``plot``](https://pyss3.readthedocs.io/en/latest/api/index.html#pyss3.util.Evaluation.plot) methods for performing tests, stratified k-fold cross validations, grid searches for hyperparameter optimization, and visualizing evaluation results using an interactive 3D plot, respectively. Probably one of its most important features is the ability to automatically (and permanently) record the history of evaluations that you've performed. This will save you a lot of time and will allow you to interactively visualize and analyze your classifier performance in terms of its different hyper-parameters values (and select the best model according to your needs). For instance, let's perform a grid search with a 4-fold cross-validation on the three [hyperparameters](https://pyss3.rtfd.io/en/latest/user_guide/ss3-classifier.html#hyperparameters), smoothness(`s`), significance(`l`), and sanction(`p`):

```python
from pyss3.util import Evaluation

best_s, best_l, best_p, _ = Evaluation.grid_search(
    clf, x_train, y_train,
    s=[0.2, 0.32, 0.44, 0.56, 0.68, 0.8],
    l=[0.1, 0.48, 0.86, 1.24, 1.62, 2],
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

In this 3D plot, each point represents an experiment/evaluation performed using that particular combination of values (`s`, `l`, and `p`). Also, these points are painted proportional to how good the performance was according to the selected metric; the plot will update "on the fly" when the user select a different evaluation metric (accuracy, precision, recall, f1, etc.). Additionally, when the cursor is moved over a data point, useful information is shown (including a "compact" representation of the confusion matrix obtained in that experiment). Finally, it is worth mentioning that, before showing the 3D plots, PySS3 creates a single and portable HTML file in your project folder containing the interactive plots. This allows users to store, send or upload the plots to another place using this single HTML file. For example, we have uploaded two of these files for you to see: ["Sentiment Analysis (Movie Reviews)"](https://pyss3.readthedocs.io/en/latest/_static/ss3_model_evaluation[movie_review_3grams].html) and ["Topic Categorization"](https://pyss3.readthedocs.io/en/latest/_static/ss3_model_evaluation[topic_categorization_3grams].html), both evaluation plots were also obtained following the [tutorials](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html#tutorials).


## Want to give PySS3 a shot? :eyeglasses: :coffee:

Just go to the [Getting Started](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html) page :D

### Installation


Simply use:
```console
pip install pyss3
```

## Want to contribute to this Open Source project? :sparkles::octocat::sparkles:

Thanks for your interest in the project, you're ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)!!
Any kind of help is very welcome (Code, Bug reports, Content, Data, Documentation, Design, Examples, Ideas, Feedback, etc.),  Issues and/or Pull Requests are welcome for any level of improvement, from a small typo to new features, help us make PySS3 better :+1:

Remember that you can use the "Edit" button ('pencil' icon) up the top to [edit any file of this repo directly on GitHub](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository).

Finally, in case you're planning to create a **new Pull Request**, for committing to this repo, we follow the "seven rules of a great Git commit message" from ["How to Write a Git Commit Message"](https://chris.beams.io/posts/git-commit/), so make sure your commits follow them as well.

(if you need any further information, please, **do not hesitate** to contact me - sergio.burdisso@gmail.com)

### Contributors :muscle::sunglasses::+1:

Thanks goes to these awesome people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="http://angermeir.me/"><img src="https://avatars3.githubusercontent.com/u/16398152?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Florian Angermeir</b></sub></a><br /><a href="https://github.com/sergioburdisso/pyss3/commits?author=angrymeir" title="Code">💻</a> <a href="#ideas-angrymeir" title="Ideas, Planning, & Feedback">🤔</a> <a href="#data-angrymeir" title="Data">🔣</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/muneebvaiyani/"><img src="https://avatars3.githubusercontent.com/u/36028992?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Muneeb Vaiyani</b></sub></a><br /><a href="#ideas-Vaiyani" title="Ideas, Planning, & Feedback">🤔</a> <a href="#data-Vaiyani" title="Data">🔣</a></td>
    <td align="center"><a href="https://www.saurabhbora.com"><img src="https://avatars2.githubusercontent.com/u/29205181?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Saurabh Bora</b></sub></a><br /><a href="#ideas-enthussb" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://hbaniecki.com"><img src="https://avatars.githubusercontent.com/u/32574004?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Hubert Baniecki</b></sub></a><br /><a href="#ideas-hbaniecki" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/sergioburdisso/pyss3/commits?author=hbaniecki" title="Documentation">📖</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Further Readings :scroll:


[Full documentation](https://pyss3.readthedocs.io)

[API documentation](https://pyss3.readthedocs.io/en/latest/api/)

[Paper preprint](https://arxiv.org/abs/1912.09322)
