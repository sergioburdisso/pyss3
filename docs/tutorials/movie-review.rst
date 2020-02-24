.. _movie-reviews:

*************************************
Sentiment Analysis (on Movie Reviews)
*************************************

In this tutorial we will develop an :ref:`SS3 classifier <ss3-classifier>` for sentiment analysis on movie reviews. We will work with a dataset called `"Large Movie Review Dataset v1.0" <https://ai.stanford.edu/~amaas/data/sentiment/>`__ (introduced in `"Learning Word Vectors for Sentiment Analysis" <https://ai.stanford.edu/~ang/papers/acl11-WordVectorsSentimentAnalysis.pdf>`__) to train our model. This dataset contains 50,000 reviews split evenly into 25k train and 25k test sets. However, since this is just a tutorial, and to speed up the whole process, we will use only a subset of it with 10K documents for training (5k pos and 5k neg) and 1k for testing (500 pos and 500 neg), and yet, as readers will experience, the final model will work remarkable well.

As it is described in :ref:`workflow`, you can choose between two possible paths to carry out this tutorial: :ref:`movie-review_classic-workflow` (using python) or :ref:`movie-review_command-line-workflow` (using only the ``PySS3 Command-Line`` tool)


.. _movie-review_classic-workflow:

Classic Workflow
================

Click :ref:`here <movie-reviews-notebook>` to go to the :ref:`tutorial notebook <movie-reviews-notebook>`.

.. note:: Want to run this Jupyter Notebook on your computer?

  1. Go to :ref:`tutorial-setup` and make sure you have everything we need.

  2. Make sure you're in the PySS3's "examples" folder...

  .. code:: console

      cd your/path/to/pyss3/examples

  and that our conda environment is activated:

  .. code:: console

      conda activate pyss3tutos

  3. Then, lunch Jupyter Notebook and and run the "movie_review.ipynb" notebook (make sure to select the "pyss3tutos" kernel).

  .. code:: console

      jupyter notebook


.. _movie-review_command-line-workflow:

Command-Line Workflow
=====================

.. note:: Before beginning, make sure you have everything ready by reading the :ref:`tutorial-setup` section.

First, make sure you're in the PySS3's "examples" folder and that our conda environment is activated:

.. code:: console

    your@user:~$ cd /your/path/to/pyss3/examples
    your@user:/your/path/to/pyss3/examples$ conda activate pyss3tutos

Make sure the dataset is unzipped, for instance by using ``unzip``:

.. code:: console

    your@user:/your/path/to/pyss3/examples$ unzip -u datasets/movie_review.zip -d datasets/


Now use the "pyss3" command to run the ``PySS3 Command Line`` tool:

.. code:: console

    your@user:/your/path/to/pyss3/examples$ pyss3

We will create a new model using the ``new`` command, we will call this model "movie_review":

.. code:: console

    (pyss3) >>> new movie_review

What are the default :ref:`hyperparameter <ss3-hyperparameter>` values? let's see

.. code:: console

    (pyss3) >>> info

which displays the following:

.. code:: console

 NAME: movie_review


 HYPERPARAMETERS:

    Smoothness(s): 0.45
    Significance(l): 0.5
    Sanction(p): 1

    Alpha(a): 0.0

 CATEGORIES: None

That is, ``s=0.45``, ``l=0.5``, and ``p=1``. Note that "CATEGORIES" is None which is OK since we haven't trained our model yet. So, let's train our model using the training set:

.. code:: console

    (pyss3) >>> train datasets/movie_review/train

Now that the model has been trained, let's see how good our model performs using the documents in the test set:

.. code:: console

    (pyss3) >>> test datasets/movie_review/test

which, among other things it displays:

.. code:: console

 accuracy: 0.853

Not bad using the default :ref:`hyperparameter <ss3-hyperparameter>` values, let's now manually analyze what our model has actually learned by using the interactive "live test".

.. code:: console

    (pyss3) >>> live_test datasets/movie_review/test

Makes sense to you? (remember you can select "words" as the Description Level if you want to know based on what words is making classification decisions)



Live test doesn't look bad, however, we will create a "more intelligent" version of this model, a version that can recognize variable-length word n-grams "on the fly". So, let's begin by creating a new model called "movie_review_3grams":

.. code:: console

    (pyss3) >>> new movie_review_3grams


As we said above, we want this model to learn to recognize variable-length n-grams, let's use the ``help`` command to see more details about the ``train`` command:

.. code:: console

    (pyss3) >>> help train

which displays the following help:

.. code:: console

        Train the model using a training set and then save it.

        usage:
            train TRAIN_PATH [LABEL] [N-gram]

        required arguments:
         TRAIN_PATH     the training set path

        optional arguments:
         LABEL          where to read category labels from.
                        values:{file,folder} (default: folder)

         N-grams        indicates the maximum n-grams to be learned (e.g. a
                        value of "1-grams" means only words will be learned;
                        "2-grams" only 1-grams and 2-grams;
                        "3-grams", only 1-grams, 2-grams and 3-grams;
                        and so on).
                        value: {N-grams} with N integer > 0 (default: 1-grams)

        examples:
         train a/training/set/path 3-grams

Yay! the ``train`` command accepts an extra argument *N*-grams (where *N* is any positive integer) that will allow us to do what we want, we will use ``3-grams`` to indicate we want SS3 to learn to recognize important words, bigrams, and 3-grams **(*)**


.. code:: console

    (pyss3) >>> train datasets/movie_review/train 3-grams

**(*)** *If you're curious and want to know how this is actually done by SS3, read the paper "t-SS3: a ext classifier with dynamic n-grams for early risk detection over text streams"* (preprint available `here <https://arxiv.org/abs/1911.06147>`__).

Now let's see if the performance has improved...

.. code:: console

    (pyss3) >>> test datasets/movie_review/test

which now displays:

.. code:: console

 accuracy: 0.856


Yeah, the accuracy slightly improved but more importantly, we should now see that the model has learned "more intelligent patterns" involving sequences of words when using the interactive "live test" to observe
what our model has learned (like "was supposed to", "has nothing to", "low budget", "your money", etc. for the "negative" class). Let's see...

.. code:: console

    (pyss3) >>> live_test datasets/movie_review/test

Finally, we will use better :ref:`hyperparameter <ss3-hyperparameter>` values. Namely, we will set ``s=0.44``, ``l=0.48`` and ``p=0.5`` which will improve the accuracy of our model:


.. code:: console

    (pyss3) >>> set s 0.44 l 0.48 p 0.5

.. note:: if you want to know how we found out that these values were going to improve our model's accuracy, it is explained in the next subsection (:ref:`hyperparameter-optimization-command-line`), so we really recommend reading it after completing this section.

Let's see if the accuracy really improves using this values:

.. code:: console

    (pyss3) >>> test datasets/movie_review/test

which displays:

.. code:: console

 accuracy: 0.861

Great! the accuracy improved :)

We will save this model in case we want to load it later...

.. code:: console

    (pyss3) >>> save

Optionally, you can again use the "live test" to manually check the final version of our model...

.. code:: console

    (pyss3) >>> live_test datasets/movie_review/test

And that's it! use the following command to exit the ``PySS3 Command Line`` (or just press Ctrl+D):

.. code:: console

    (pyss3) >>> exit

Congratulations! you have created an SS3 model for sentiment analysis without a single line of code, buddy :)

.. _hyperparameter-optimization-command-line:

Hyperparameter Optimization
----------------------------

As mentioned earlier, hyperparameter optimization will allow us to find better :ref:`hyperparameter <ss3-hyperparameter>` values for our model.  To begin with, we will perform a grid search over the test set. To carry out this task, we will use the ``grid_search`` command. Let's see what this command does and how to use it, using the ``help`` command:

.. code:: console

    (pyss3) >>> help grid_search

which displays the following help:

.. code:: console

    Given a dataset, perform a grid search using the given hyperparameters values.

    usage:
        grid_search PATH [LABEL] [DEF_CAT] [METHOD] P EXP [P EXP ...] [no-cache]

    required arguments:
     PATH       the dataset path
     P EXP      a list of values for a given hyperparameter.
                where:
                 P    is a hyperparameter name. values: {s,l,p,a}
                 EXP  is a python expression returning a float or
                      a list of floats. Note: if this expression
                      contains whitespaces, use quotations marks
                      (e.g. "[0.5, 1.5]")
                examples:
                 s [.3,.4,.5]
                 s "[.3, .4, .5]" (Note the whitespaces and the "")
                 p r(.2,.8,6)     (i.e. 6 points between .2 to .8)

    optional arguments:
     LABEL      where to read category labels from.
                values:{file,folder} (default: folder)

     DEF_CAT    default category to be assigned when the model is not
                able to actually classify a document.
                values: {most-probable,unknown} or a category label
                (default: most-probable)

     METHOD     the method to be used
                values: {test, K-fold} (default: test)
                where:
                  K-fold  indicates the number of folds to be used.
                          K is an integer > 1 (e.g 4-fold, 10-fold, etc.)

     no-cache   if present, disable the cache and recompute all the values

    examples:
     grid_search a/testset/path s r(.2,.8,6) l r(.1,2,6) -p r(.5,2,6) a [0,.01]
     grid_search a/dataset/path 4-fold -s [.2,.3,.4,.5] -l [.5,1,1.5] -p r(.5,2,6)

From this help, we can see that this command expects at least a path and a list of :ref:`hyperparameter <ss3-hyperparameter>` names and, after each :ref:`hyperparameter <ss3-hyperparameter>` name, any python expression that returns either a number or a list of numbers, for instance, ``-s [.2,.35,.4,.55]``. In our case, we will use the built-in function ``r(x0,x1,n)`` which returns a list of ``n`` numbers between ``x0`` and ``x1`` (including both), as follows:

.. code:: console

    (pyss3) >>> grid_search datasets/movie_review/test -s r(.2,.8,6) -l r(.1,2,6) -p r(.5,2,6)

*Note that here,* ``s`` *will take 6 different values between .2 and .8,* ``l`` *between .1 and 2, and* ``p`` *between .5 and 2.*

Now it is time to wait (for about 20 minutes) until the grid search is completed.

Once the grid search is over, we will use the following command to open up an interactive 3D plot in the browser that we can use to analyze the obtained results:

.. code:: console

    (pyss3) >>> plot evaluations


PySS3 should have created `this plot <../_static/ss3_model_evaluation[movie_review_3grams].html>`__ on your machine. **Note:** We recommend reading the :ref:`evaluation-plot` page in which the plots and the user interface are explained in detail.

You probably noted that there are multiple points with the global best performance, this is probably due to this problem (sentiment analysis) being a binary classification problem (thus, the "sanction" :ref:`hyperparameter <ss3-hyperparameter>` doesn't have much impact with only two categories).  We could choose any of the best values, for instance, we will select the one with the lowest "sanction" (p) value. To do this, rotate the plot and move the cursor over this point and see the information that is displayed, as shown in the following figure:

.. image:: ../_static/movie_review_evaluations.png

Here we can see that using these :ref:`hyperparameter <ss3-hyperparameter>` values, our classifier will obtain a better accuracy (0.861):

* smoothness (:math:`\sigma`): 0.44
* significance (:math:`\lambda`): 0.48
* sanction (:math:`\rho`): 0.5

That is, we need to set ``s=0.44``, ``l=0.48`` and ``p=0.5``. To do this we could use the ``set`` and ``save`` commands to update and save our model for later use:

.. code:: console

    (pyss3) >>> set s 0.44 l 0.48 p 0.5
    (pyss3) >>> save


.. note::
  if you want to use these hyperparameter values with python, there are at least three ways we can configure our SS3 classifier:

  * Creating a new classifier using these hyperparameter values:

  .. code:: python

      clf = SS3(s=0.44, l=0.48, p=0.5)


  * Changing the hyperparameter values of an already existing model using the ``set_hyperparameters`` method:

  .. code:: python

      clf = SS3()
      ...
      clf.set_hyperparameters(s=0.44, l=0.48, p=0.5)


  * Or, using the ``PySS3 Command Line``:

      1. Use the ``set`` and ``save`` commands to update and save the model

      .. code:: console

          (pyss3) >>> set s 0.44 l 0.48 p 0.5
          (pyss3) >>> save

      2. And then, use the ``load_model`` method to load the model with python:

      .. code:: python

          clf = SS3(name="movie_review_3grams")
          ...
          clf.load_model()

Before we finish the hyperparameter optimization task, there is an optional (but recommended) step. To make sure the selected :ref:`hyperparameters <ss3-hyperparameter>` generalize well (i.e. are not overfitted to the test set), we will perform an extra grid search but this time using a (stratified) 10-fold cross-validation. From what we saw from the previous grid search, the "santion"(p) hyperparameter doesn't seem to have a real impact on performance, so this time we will set ``p = 0.5`` when performing the grid search, that is:

.. code:: console

    (pyss3) >>> grid_search datasets/movie_review/train 10-fold -s r(.2,.8,6) -l r(.1,2,6) -p 0.5

This grid search will take about 40 minutes to complete, I know, it may seem like a lot but remember that, since we are using 10-fold cross-validation, for each hyperparameter value combination we have to train and test the model 10 times!

When the search is over, use once again the ``plot`` command: 

.. code:: console

    (pyss3) >>> plot evaluations

Now, using the options panel change "Tag" option and select the path we used for this last grid search ("datasets/movie_review/train"), as shown in the following image:

.. image:: ../_static/movie_review_evaluations_kfold_op.png

Fortunately, the same point we have previously selected has also the best performance here:  

.. image:: ../_static/movie_review_evaluations_kfold.png

Note that all the 10 confusion matrices looks really well and consistent, that means that this configuration performed consistently well across the 10 different folds! this means we can use the selected :ref:`hyperparameter <ss3-hyperparameter>` values (``s=0.44``, ``l=0.48`` and ``p=0.5``) safely.

(Feel free to play a little bit with this interactive 3D evaluation plot, for instance try changing the metric and target from the options panel)