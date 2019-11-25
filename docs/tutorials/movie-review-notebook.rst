.. _movie-reviews-notebook:

Movie Review (Sentiment Analysis) - Jupyter Notebook
====================================================

This is the static (html) version of the notebook for the tutorial :ref:`"Movie Review - Classic
Workflow" <movie-review_classic-workflow>`.

(the orginal notebook file, "movie_review.ipynb", can be found `here <https://github.com/sergioburdisso/pyss3/tree/master/examples>`__, and an interactive online version `here <https://mybinder.org/v2/gh/sergioburdisso/pyss3/master?filepath=examples/movie_review.ipynb>`__)

--------------

Before we begin, let's import needed modules...

.. code:: python

    from pyss3 import SS3
    from pyss3.util import Dataset
    from pyss3.server import Server
    
    from sklearn.metrics import accuracy_score

... and unzip the "movie\_review.zip" dataset inside the ``datasets``
folder.

.. code:: shell

    !unzip -u datasets/movie_review.zip -d datasets/


Ok, now we are ready to begin. Let's create a new SS3 instance

.. code:: python

    clf = SS3()

What are the default :ref:`hyperparameter <ss3-hyperparameter>` values? let's see

.. code:: python

    s, l, p, _ = clf.get_hyperparameters()
    
    print("Smoothness(s):", s)
    print("Significance(l):", l)
    print("Sanction(p):", p)


.. parsed-literal::

    Smoothness(s): 0.45
    Significance(l): 0.5
    Sanction(p): 1


Ok, now let's load the training and the test set using the
``load_from_files`` function from ``pyss3.util`` as follow:

.. code:: python

    x_train, y_train = Dataset.load_from_files("datasets/movie_review/train")
    x_test, y_test = Dataset.load_from_files("datasets/movie_review/test")


Let's train our model...

.. code:: python

    clf.fit(x_train, y_train)


.. parsed-literal::

     Training: 100%|██████████| 2/2 [00:13<00:00,  6.51s/it]

Note that we don't have to create any document-term matrix! we are using
just the plain ``x_train`` documents :D cool uh? (SS3 creates a language
model for each category and therefore it doesn't need to create any
document-term matrices)


Now that the model has been trained, let's test it using the documents
in ``x_test``

.. code:: python

    y_pred = clf.predict(x_test)

.. parsed-literal::

     Classification: 100%|██████████| 1000/1000 [00:04<00:00, 200.12it/s]

Let's see how good our model performed

.. code:: python

    print("Accuracy:", accuracy_score(y_pred, y_test))


.. parsed-literal::

    Accuracy: 0.852


Not bad using the default :ref:`hyperparameter <ss3-hyperparameter>` values, let's now manually
analyze what our model has actually learned by using the interactive
"live test".

.. code:: python

    Server.serve(clf, x_test, y_test)

Makes sense to you? (remember you can select "words" as the
Description Level if you want to know based on what words is making
classification decisions)

Live test doesn't look bad, however, we will create a "more intelligent"
version of this model, a version that can recognize variable-length word
n-grams "on the fly". Thus, when calling the ``fit`` we will pass an
extra argument ``n_grams=3`` to indicate we want SS3 to learn to
recognize important words, bigrams, and 3-grams **[*]**. Additionally, we will name our model "movie\_review\_3grams" so that we can save it and load it later from the ``PySS3 Command Line`` to perform
the hyperparameter optimization to find better :ref:`hyperparameter <ss3-hyperparameter>` values.

**[*]** *If you're curious and want to know how this is actually
done by SS3, read the paper "t-SS3: a text classifier with dynamic
n-grams for early risk detection over text streams"* (preprint available
`here <https://arxiv.org/abs/1911.06147>`__).

.. code:: python

    clf = SS3(name="movie_review_3grams")
    
    clf.fit(x_train, y_train, n_grams=3)  # <-- note the n_grams=3 argument here


.. parsed-literal::

     Training: 100%|██████████| 2/2 [00:19<00:00, 10.00s/it]




As mentioned above, we will save this trained model for later use.

.. code:: python

    clf.save_model()


.. parsed-literal::

    [ saving model (ss3_models/movie_review_3grams.ss3m)... ]


Now let's see if the performance has improved...

.. code:: python

    y_pred = clf.predict(x_test)


.. parsed-literal::

     Classification: 100%|██████████| 1000/1000 [00:05<00:00, 195.64it/s]


.. code:: python

    print("Accuracy:", accuracy_score(y_pred, y_test))


.. parsed-literal::

    Accuracy: 0.855


Yeah, the accuracy slightly improved but more importantly, we should now
see that the model has learned "more intelligent patterns" involving
sequences of words when using the interactive "live test" to observe
what our model has learned (like "was supposed to", "has nothing to",
"low budget", "your money", etc. for the "negative" class). Let's see...

.. code:: python

    Server.serve(clf, x_test, y_test)


.. _movie-review-notebook-continue:

**Before moving forward, at this point you should read the** :ref:`hyperparameter-optimization` **section of this tutorial.**

As described in the "Hyperparameter Optimization" section, after performing hyperparameter
optimization using the ``PySS3 Command Line``, we found out that, for
example, the following :ref:`hyperparameter <ss3-hyperparameter>` values will slightly improve our
classification performance

.. code:: python

    clf.set_hyperparameters(s=.44, l=.48, p=1.1)

Let's see if it's true...

.. code:: python

    y_pred = clf.predict(x_test)


.. parsed-literal::

     Classification: 100%|██████████| 1000/1000 [00:06<00:00, 148.17it/s]


.. code:: python

    print("Accuracy:", accuracy_score(y_pred, y_test))


.. parsed-literal::

    Accuracy: 0.861


Great! accuracy improved. Fortunately, this time we got lucky and the
default hyperparameters were also quite good.
