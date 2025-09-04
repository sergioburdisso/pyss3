<img src="https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/ss3_logo_banner.png" alt="PySS3 Logo" title="PySS3" height="150" />

[![Documentation Status](https://readthedocs.org/projects/pyss3/badge/?version=latest)](http://pyss3.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://api.travis-ci.com/sergioburdisso/pyss3.svg?branch=master)](https://app.travis-ci.com/github/sergioburdisso/pyss3)
[![codecov](https://codecov.io/gh/sergioburdisso/pyss3/branch/master/graph/badge.svg)](https://codecov.io/gh/sergioburdisso/pyss3)
[![PyPI version](https://badge.fury.io/py/pyss3.svg)](https://badge.fury.io/py/pyss3)
[![Downloads](https://static.pepy.tech/badge/pyss3)](https://pepy.tech/project/pyss3)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sergioburdisso/pyss3/master?filepath=examples)

---

# PySS3: Interpretable Machine Learning for Text Classification (Try our [**Live Demo**](https://sergioburdisso.github.io/ss3/)! :cake:)

<br>

PySS3 implements **SS3**, a simple supervised machine learning model for **interpretable text classification**. SS3 can **self-explain its rationale**, making it a reliable choice for tasks where understanding model decisions is critical.  

It was originally introduced in Section 3 of _["A text classification framework for simple and effective early depression detection over social media streams"](https://dx.doi.org/10.1016/j.eswa.2019.05.023)_ ([arXiv preprint](https://arxiv.org/abs/1905.08772)) and obtained the best and second-best results, consecutively, in the three [CLEF eRisk](https://erisk.irlab.org/) editions from 2019 to 2021 [[Burdisso *et al.* 2019](http://ceur-ws.org/Vol-2380/paper_103.pdf); [Loyola *et al.* 2021](http://ceur-ws.org/Vol-2936/paper-81.pdf)].

PySS3 also includes **variants of SS3**, such as _t-SS3_, which dynamically recognizes variable-length word n-grams "on the fly" for early risk detection ([paper](https://doi.org/10.1016/j.patrec.2020.07.001), [arXiv](https://arxiv.org/abs/1911.06147)).

---

## What is PySS3?

[PySS3](https://github.com/sergioburdisso/pyss3) is a **Python library for working with SS3 in a visual, interactive, and straightforward way**.  

It provides tools to:

- Analyze, monitor, and understand what your model has learned.  
- Visualize classification decisions and model insights.  
- Evaluate and optimize hyperparameters efficiently.  

The library is organized into **three main components**:

---

### :point_right: ``SS3`` class

The core classifier with a clean API similar to `sklearn`:

```python
from pyss3 import SS3

clf = SS3()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
```

Other useful methods include:

- [`extract_insight()`](https://pyss3.rtfd.io/en/latest/tutorials/extract-insight.html) – Returns a **list of text fragments** involved in the classification decision, allowing you to understand the rationale behind the model’s predictions.  
- [`classify_multilabel()`](https://pyss3.rtfd.io/en/latest/api/index.html#pyss3.SS3.classify_multilabel) – Multi-label classification support:

```python
doc = "Liverpool CEO Peter Moore on Building a Global Fanbase"

label = clf.classify_label(doc)          # 'business'
labels = clf.classify_multilabel(doc)    # ['business', 'sports']
```

See [all tutorials](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html#tutorials) for step-by-step guidance.

---

### :point_right: ``Live_Test`` class

Interactively test models in your browser, **with one line of code**:

```python
from pyss3.server import Live_Test

Live_Test.run(clf, x_test, y_test)
```

![img](https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/ss3_live_test.gif)

Try our **online live demos**:

- [Movie Review (Sentiment Analysis)](https://sergioburdisso.github.io/ss3/live_test_online/#30305)  
- [Topic Categorization](https://sergioburdisso.github.io/ss3/live_test_online/#30303)  

---

### :point_right: `Evaluation` class

Evaluate and optimize your model easily:

```python
from pyss3.util import Evaluation

best_s, best_l, best_p, _ = Evaluation.grid_search(
    clf, x_train, y_train,
    s=[0.2, 0.32, 0.44, 0.56, 0.68, 0.8],
    l=[0.1, 0.48, 0.86, 1.24, 1.62, 2],
    p=[0.5, 0.8, 1.1, 1.4, 1.7, 2],
    k_fold=4
)
Evaluation.plot()
```

- Interactive 3D plots for hyperparameter evaluation  
- Automatic history tracking of experiments  
- Exportable HTML plots for sharing and reporting  

![img](https://raw.githubusercontent.com/sergioburdisso/pyss3/master/docs/_static/plot_evaluations.gif)

Explore example evaluation plots:

- [Sentiment Analysis (Movie Reviews)](https://pyss3.readthedocs.io/en/latest/_static/ss3_model_evaluation[movie_review_3grams].html)  
- [Topic Categorization](https://pyss3.readthedocs.io/en/latest/_static/ss3_model_evaluation[topic_categorization_3grams].html)  

---

## Getting Started :eyeglasses: :coffee:

### Installation

```bash
pip install pyss3
```

[Full tutorial and documentation](https://pyss3.readthedocs.io/en/latest/user_guide/getting-started.html)  

---

## Contributing :sparkles::octocat::sparkles:

Any contributions are welcome! Code, bug reports, documentation, examples, or ideas – everything helps.  

Use the "Edit" button on GitHub to propose changes directly, and follow [these guidelines](https://chris.beams.io/posts/git-commit/) for commit messages.

---

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
