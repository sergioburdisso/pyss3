.. _movie-reviews-notebook:

Movie Review (Sentiment Analysis) Jupyter Notebook
==================================================

This is the notebook for the tutorial `"Movie Review - Classic
Workflow" <https://pyss3.readthedocs.io/en/latest/tutorials/movie-review.html#classic-workflow>`__
of `PySS3 <https://pyss3.readthedocs.io>`__.

(the real notebook can be found `here <https://github.com/sergioburdisso/pyss3/tree/master/examples>`__)

--------------

Before we begin, let's import needed modules...

.. code:: python

    from pyss3 import SS3
    from pyss3.util import Dataset
    from pyss3.server import Server
    
    from sklearn.metrics import accuracy_score
    
    import zipfile

... and unzip the "movie\_review.zip" dataset inside the ``datasets``
folder.

.. code:: python

    !unzip -u datasets/movie_review.zip -d datasets/

Ok, now we are ready to begin. Let's create a new SS3 instance. We will
name it "movie\_review" so that we can load the trained model from the
``PySS3 Command Line`` and perform a hyper-parameter optimization to
find the best hyper-parameters values.

.. code:: python

    clf = SS3(name="movie_review")

What are the default hyper-parameter values? let's see

.. code:: python

    s, l, p, _ = clf.get_hyperparameters()
    
    print("Smoothness(s):", s)
    print("Significance(l):", l)
    print("Sanction(p):", p)

Ok, now let's load the training and the test set using the
``load_from_files`` function from ``pyss3.util`` as follow:

.. code:: python

    x_train, y_train = Dataset.load_from_files("datasets/movie_review/train")
    x_test, y_test = Dataset.load_from_files("datasets/movie_review/test")

Let's train our model...

.. code:: python

    clf.fit(x_train, y_train)

Additionally, we will save this trained model for later use (from the
``PySS3 Command Line`` as the tutorial suggest)

.. code:: python

    clf.save_model()

Now that the model has been trained, let's test it using the documents
in ``x_test``

.. code:: python

    y_pred = clf.predict(x_test)

Let's see how good our model performed

.. code:: python

    print("Accuracy:", accuracy_score(y_pred, y_test))

Not bad using the default hyper-parameters values, let's now manually
analyze what our model has actually learned by using the interactive
"live test". Makes sense to you?

.. code:: python

    Server.serve(clf, x_test, y_test)

(!) Press ``Esc`` key and then the ``I`` key twice to stop the server

*At this point you should go back to the tutorial page so that you can
learn how to use the ``PySS3 Command Line`` for model selection, once
you've completed that part continue with the following paragraph.*

As described in the tutorial, after performing hyper-parameters
optimization using the ``PySS3 Command Line``, we found out that, for
example, the following hyper-parameter values will slightly improve our
classification performance

.. code:: python

    clf.set_hyperparameters(s=.44, l=.48, p=1.1)

Let's see if it's true...

.. code:: python

    y_pred = clf.predict(x_test)

.. code:: python

    print("Accuracy:", accuracy_score(y_pred, y_test))

Great! fortunately, we got lucky and the default hyper-parameters were
quite good!

What? Want to try this slightly better model? Ok, let's use the PySS3
server again :)

.. code:: python

    Server.serve(clf, x_test, y_test)
