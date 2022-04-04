---
title: 'PySS3: A Python package implementing SS3, a simple and interpretable machine learning model for text classification'
tags:
  - Python
  - Machine learning
  - Natural Language Processing
  - Text Classification
  - Interpretability
  - Explainable Artificial Intelligence
authors:
  - name: Sergio G. Burdisso^[corresponding author]
    # orcid: 0000-000...
    affiliation: "1, 2"
  - name: Marcelo Errecalde
    affiliation: 1
  - name: Manuel Montes-y-Gómez
    affiliation: 3
affiliations:
 - name: Universidad Nacional de San Luis (UNSL), Argentina
   index: 1
 - name: Consejo Nacional de Investigaciones Científicas y Técnicas (CONICET), Argentina
   index: 2
 - name: Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE), México
   index: 3
date: 15 October 2021
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary


In this paper, we briefly introduce `PySS3`[^pyss3] and share it with the community. `PySS3` is an open-source Python package that implements the SS3 machine learning model for text classification. `PySS3` comes with useful tools that allow working with SS3 in an interactive, and visual way. For instance, one of these tools provides post hoc explanations using visualizations that directly highlight relevant portions of the raw input document, allowing researchers to better understand the models being deployed. Therefore, `PySS3` could be especially useful for those working with sensitive classification problems by which people's lives could be affected since it allows researchers and practitioners to deploy interpretable (i.e. self-explainable) and more reliable models for text classification. 

[^pyss3]: [https://github.com/sergioburdisso/pyss3](https://github.com/sergioburdisso/pyss3)

# Statement of need

A challenging scenario in the machine learning field is the one referred to as _early classification_. Early classification deals with the problem of classifying data streams as early as possible without having a significant loss in performance.
The reasons behind this requirement of _earliness_ could be diverse, but the most important and interesting case is when the classification delay has negative or risky implications.
This scenario, known as _Early Risk Detection_ (ERD), has gained increasing interest in recent years with potential applications in rumor detection [@ma2015detect;@ma2016detecting;@kwon2017rumor], sexual predator detection, aggressive text identification [@escalante2017early], depression detection [@losada2017erisk;@losada2016test], and terrorism detection [@iskandar2017terrorism], among others.

A recently introduced machine learning model for text classification [@burdisso2019;@burdisso2019-tss3], called SS3, has shown to be well suited to deal with ERD problems on social media streams.
It obtained state-of-the-art performance on early depression, anorexia and self-harm detection on the latest CLEF's eRisk lab challenges [@loyola2021unsl;@burdisso2019clef;@burdisso2019;@burdisso2019-tss3].
Unlike standard classifiers, this new classification model was specially designed to deal with ERD problems since: (a) it is interpretable and therefore can naturally self-explain its rationale, and (b) it naturally supports incremental training and classification over text streams.

However, little attention has been paid to the potential use of SS3 as a general-purpose classifier for other text classification tasks.
One of the main reasons could be the fact that there is no open-source implementation of SS3 available yet.
We believe that the availability of open-source implementations is of critical importance to foster the use of new tools, methods, and algorithms.
On the other hand, Python has come to be the most popular programming language in the machine learning community thanks to its simple syntax and a rich ecosystem of efficient open-source implementations of popular algorithms [@python2007].
Therefore, we decided to develop an open-source Python package to provide the first official implementation of this new classifier.


# Implementation

`PySS3` was coded to be compatible with Python 2 and Python 3 as well as with different operating systems, such as Linux, macOS, and Microsoft Windows. To ensure this compatibility holds whenever the source code is updated, we have configured and linked the Github repository with the Travis CI service. This service automatically runs the test scripts using different operating systems and versions of Python whenever new code is pushed to the repository.

The package is composed of one main module and three submodules.[^more_info] 
The main module is called `pyss3` and contains the classifier's implementation _per se_ in a class called `SS3`.
This class not only implements the "plain-vanilla" version of the classifier [@burdisso2019] but also different variations, such as the one introduced later by Burdisso _et al._ [@burdisso2019-tss3], which allows SS3 to recognize important word n-grams on the fly.
As the reader will notice in the example shown in the next section, this class exposes a clear and simple API that is similar to that of _Scikit-learn_ models [@pedregosa2011scikit]. For instance, it provides standard methods like `fit()` and `predict()` for  training and classification, respectively.[^api_doc]
Finally, the three submodules, `pyss3.server`, `pyss3.cmd_line`, and `pyss3.util`, provide a collection of useful tools and utility functions such as, for instance, those related to data loading, evaluation or, as will be shown in the next section,  "live" testing the models.

[^more_info]: A more detailed description of the package is given in the official documentation ([https://pyss3.rtfd.io](https://pyss3.rtfd.io)). Additionally, an extended version of this paper can be found in ArXiv[@burdisso2019pyss3].

[^api_doc]: [https://pyss3.rtfd.io/en/latest/api](https://pyss3.rtfd.io/en/latest/api/index.html\#pyss3.SS3)


# Illustrative examples

In this section, we will only introduce two simple illustrative examples.
For full and real working examples, please refer to the [tutorials](https://pyss3.rtfd.io/en/latest/user_guide/getting-started.html\#tutorials) in the documentation.
Additionally, for readers interested in trying PySS3 out "right away", we have created Jupyter Notebooks for the tutorials,[^note] which can be executed online, for instance, using [`MyBinder`](https://mybinder.org/v2/gh/sergioburdisso/pyss3/master?filepath=examples).

[^note]: [https://github.com/sergioburdisso/pyss3/tree/master/examples](https://github.com/sergioburdisso/pyss3/tree/master/examples)

Before introducing the examples, we first need to load the training and test documents and category labels, as usual, in the `x_train`, `y_train`, `x_test`, `y_test` lists, respectively.
For instance, we can use the [`load_from_url()`](https://pyss3.rtfd.io/en/latest/api/index.html#pyss3.util.Dataset.load_from_url) function from `Dataset` class to  load the ["Topic Categorization"](https://pyss3.rtfd.io/en/latest/tutorials/topic_categorization-notebook.html) tutorial's dataset, as follows:

````python
from pyss3.util import Dataset

url = "https://github.com/sergioburdisso/pyss3/raw/master/examples/datasets/topic.zip"

x_train, y_train = Dataset.load_from_url(url, "train", folder_label=False)
x_test, y_test = Dataset.load_from_url(url, "test", folder_label=False)
````

This dataset was created collecting about 30k tweets and contains the following 8 different class labels: _"art&photography", "beauty&fashion", "business&finance", "food", "health", "music", "science&technology"_, and _"sports"_.

## Training and test example

This simple example shows how to train and test an SS3 model using default values. Since SS3 creates a language model for each category, we do not need to create a document-term matrix, we can simply use the raw `x_train` and `x_test` documents for training and test, respectively, as follows:

````python
from pyss3 import SS3

clf = SS3()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy:", accuracy(y_pred, y_test))
````


## Training and live test example

This example is similar to the previous one. However, instead of simply using `predict` and `accuracy` to measure our model's performance, here we are using the `PySS3`'s "Live Test" tool. 
The "Live Test" is an interactive visualization tool that allows users to test the models "on the fly" with a single line of extra code, as follows:

````python
from pyss3 import SS3
from pyss3.server import Live_Test

clf = SS3()
clf.fit(x_train, y_train)

# The following line will open up the Live Test tool, shown in Figure 1
Live_Test.run(clf, x_test, y_test)
````

As shown in \autoref{fig:live_test}, the tool provides a user interface by which users can manually and actively test their model using either the documents in the test set or just typing in their own. More precisely, this tool allows researchers to analyze and understand what their models are learning by providing an interactive and visual explanation of the classification process at three different levels (word n-grams, sentences, and paragraphs).^[We recommend trying out the "Topic Categorization" or the "Sentiment Analysis on Movie Reviews" online live demo available at [http://tworld.io/ss3](http://tworld.io/ss3)]

Alternatively to the Live Test tool, we could also use the `clf.extract_insight()` function. When this function is applied to a document, it returns the list of text fragments involved in the classification decision, ordered by a confidence value. For instance, suppose we need to obtain the text fragment with the highest confidence value that was used to classify the document shown in \autoref{fig:live_test} as "sports," we could use `extract_insight()` as follows:

````python
# Assign to "doc" the same document that is shown in Figure 1
doc = "Last year, Moore became Liverpool's CEO. This season, his club ..."

fragments = clf.extract_insight(doc, "sports")

print(fragments[0])
````

Which will print the following (fragment, confidence value) pair:

````python
('to the Champions League Final. (Liverpool will play Real Madrid in
Kiev, Ukraine on Saturday). Meanwhile Moore has',  0.97)
````

This pair indicates the document was classified as "sports", mainly due to our model finding this fragment as belonging to "sports" with a confidence value of 0.97.^[Interested readers may refer to the "getting the text fragments involved in the classification decision" tutorial [available online](https://pyss3.rtfd.io/en/latest/tutorials/extract-insight.html)]

![Live Test screenshot. On the left side, the list of test documents grouped by category is shown along with the percentage of success (true positive ratio).
Note the `doc_2` document is marked with an exclamation mark (!); this mark indicates it was misclassified, which eases error analysis.
The user has selected the `doc_1`, the "classification result" is shown above the visual description. 
In this figure, the user has chosen to display the visual explanation at sentence-and-word level, using mixed topics.
For instance, the user can confirm that, apparently, the model has learned to recognize important words and that it has correctly classified the document.
Also, by using the colors, the user could notice that the first sentence belonging to multiple topics, the second sentence shifted the topic to $sports$, and finally, from "Meanwhile" on, the topic is shifted to `technology` (and a little bit of `business` given by the words "investment" or "engage" colored in green).
Note that the user can also edit the document or even create new ones using the two buttons on the upper-right corner.\label{fig:live_test}](images/fig1.jpg)

# Conclusions

We have briefly introduced `PySS3`, an open-source Python package that implements SS3, an interpretable machine learning model for text classification. As such, PySS3 comes with useful visualization tools that help understand the reasons behind its classification decisions.
This software could be especially useful for researchers and practitioners interested in deploying interpretable and more reliable models for text classification.
Finally, we hope to continue the advancement and improvement of PySS3 through the direct or indirect help of those in the mathematical and computer science disciplines who want to be part of this open-source project.


# References