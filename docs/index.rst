.. image:: _static/ss3_logo_banner.png
    :target: https://github.com/sergioburdisso/pyss3
    :height: 100px
    :align: right


Welcome to PySS3's documentation!
=================================


`PySS3`_ is a Python package that allows you to work with :ref:`ss3-classifier` in a very
straightforward, interactive and visual way. In addition to the
implementation of the classifier, PySS3 comes with a set of tools
to help you developing your machine learning models in a clearer and
faster way. These tools let you analyze, monitor and understand your
models by allowing you to see what they have actually learned and why. To
achieve this, PySS3 provides you with 3 main components: the ``SS3``
class, the ``Live_Test`` class and the ``PySS3 Command Line`` tool, as
pointed out below.


The ``SS3`` class
~~~~~~~~~~~~~~~~~

which implements the classifier using a clear API (very similar to that
of ``sklearn``):

.. code:: python

        from pyss3 import SS3
        clf = SS3()
        ...
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)


The ``Live_Test`` class
~~~~~~~~~~~~~~~~~~~~

which allows you to interactively test your model and visually see the
reasons behind classification decisions, **with just one line of code**:

.. code:: python

        from pyss3.server import Live_Test
        from pyss3 import SS3

        clf = SS3(name="my_model")
        ...
        clf.fit(x_train, y_train)
        Live_Test.run(clf, x_test, y_test) # <- this one! cool uh? :)

As shown in the image below, this will open up, locally, an interactive
tool in your browser which you can use to (live) test your models with
the documents given in ``x_test`` (or typing in your own!). This will
allow you to visualize and understand what your model is actually
learning.

.. image:: _static/ss3_live_test.gif

For example, we have uploaded two of these live tests online for you to
try out: `"Movie Review (Sentiment Analysis)" <http://tworld.io/ss3/live_test_online/#30305>`__ and
`"Topic
Categorization" <http://tworld.io/ss3/live_test_online/#30303>`__, both
were obtained following the :ref:`tutorials`.


And last but not least, the ``PySS3 Command Line`` tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is probably the most useful component of PySS3. When you install
the package (for instance by using ``pip install pyss3``) a new command
``pyss3`` is automatically added to your environment's command line.
This command allows you to access to the *PySS3 Command Line*, an
interactive command-line query tool. This tool will let you interact
with your SS3 models through special commands while assisting you during
the whole machine learning pipeline (model selection, training, testing,
etc.). Probably one of its most important features is the ability to
automatically (and permanently) record the history of every evaluation
result of any type (tests, k-fold cross-validations, grid searches,
etc.) that you've performed. This will allow you (with a single command)
to interactively visualize and analyze your classifier performance in
terms of its different hyperparameters values (and select the best
model according to your needs). For instance, let's perform a grid
search with a 4-fold cross-validation on the three :ref:`hyperparameters <ss3-hyperparameter>`,
smoothness(\ ``s``), significance(\ ``l``), and sanction(\ ``p``) as
follows:

.. code:: console

    your@user:/your/project/path$ pyss3
    (pyss3) >>> load my_model
    (pyss3) >>> grid_search path/to/dataset 4-fold -s r(.2,.8,6) -l r(.1,2,6) -p r(.5,2,6)

In this illustrative example, ``s`` will take 6 different values between
0.2 and 0.8, ``l`` between 0.1 and 2, and ``p`` between 0.5 and 2. After the
grid search finishes, we can use the following command to open up an
interactive 3D plot in the browser:

.. code:: console

    (pyss3) >>> plot evaluations

.. image:: _static/plot_evaluations.gif

Each point represents an experiment/evaluation performed using that
particular combination of values (s, l, and p). Also, these points are painted
proportional to how good the performance was using that configuration of
the model. Researchers can interactively change the evaluation metrics
to be used (accuracy, precision, recall, f1, etc.) and plots will update
"on the fly". Additionally, when the cursor is moved over a data point,
useful information is shown (including a "compact" representation of the
confusion matrix obtained in that experiment). Finally, it is worth
mentioning that, before showing the 3D plots, PySS3 creates a
single and portable HTML file in your project folder containing the
interactive plots. This allows researchers to store, send or upload the
plots to another place using this single HTML file (or even provide a
link to this file in their own papers, which would be nicer for readers,
plus it would increase experimentation transparency). For example, we
have uploaded two of these files for you to see: `"Movie Review (Sentiment Analysis)" <_static/ss3_model_evaluation[movie_review_3grams].html>`__
and `"Topic
Categorization" <_static/ss3_model_evaluation[topic_categorization_3grams].html>`__,
both evaluation plots were obtained following the :ref:`tutorials`.


Want to give PySS3 a try?
=========================

Just go to the :ref:`getting-started` page :D

----

Further Readings
================

.. toctree::
   :maxdepth: 3
   :caption: User Guide

   user_guide/getting-started
   user_guide/installation
   user_guide/workflow
   user_guide/ss3-classifier
   user_guide/visualizations
..   user_guide/customization
..   user_guide/pyss3-command-line


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/setup
   tutorials/topic-categorization
   tutorials/movie-review

.. toctree::
   :maxdepth: 3
   :caption: API Documentation

   api/index

.. toctree::
   :maxdepth: 2
   :caption: About

   about/changelog
   about/contributing
   about/license

.. _PySS3: https://github.com/sergioburdisso/pyss3
.. _“Movie Review (Sentiment Analysis)”: http://tworld.io/ss3/live_test_online/#30305
.. _“Topic Categorization”: http://tworld.io/ss3/live_test_online/#30303
