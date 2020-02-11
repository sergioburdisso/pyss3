# coding: utf-8
"""
Movie Review (Sentiment Analysis) Tutorial Source Code (Classic Workflow).

(https://pyss3.readthedocs.io/en/latest/tutorials/movie-review.html#movie-review_classic-workflow)
---
"""

# Before we begin, let's import needed modules...
from pyss3 import SS3
from pyss3.util import Dataset
from pyss3.server import Live_Test

from sklearn.metrics import accuracy_score
from os import system


# ... and unzip the "movie_review.zip" dataset inside the `datasets` folder.
system('unzip -u datasets/movie_review.zip -d datasets/')


# Ok, now we are ready to begin. Let's create a new SS3 instance.
clf = SS3()

# What are the default hyper-parameter values? let's see
s, l, p, _ = clf.get_hyperparameters()

print("Smoothness(s):", s)
print("Significance(l):", l)
print("Sanction(p):", p)

# Ok, now let's load the training and the test set using the `load_from_files`
# function from `pyss3.util` as follow:
x_train, y_train = Dataset.load_from_files("datasets/movie_review/train")
x_test, y_test = Dataset.load_from_files("datasets/movie_review/test")

# Let's train our model...
clf.fit(x_train, y_train)

# Note that we don't have to create any document-term matrix! we are using just
# the plain `x_train` documents :D cool uh?
# (SS3 creates a language model for each category and therefore it doesn't need
# to create any document-term matrices)

# Now that the model has been trained, let's test it using the documents in `x_test`
y_pred = clf.predict(x_test)

# Let's see how good our model performed
print("Accuracy:", accuracy_score(y_pred, y_test))

# Not bad using the default hyper-parameters values, let's now manually
# analyze what our model has actually learned by using the interactive "live test".
Live_Test.run(clf, x_test, y_test)

# Makes sense to you?
# (remember you can select "words" as the Description Level if you want to know
# based on what words is making classification decisions)


# Live test doesn't look bad, however, we will create a "more intelligent" version of
# this model, a version that can recognize variable-length word n-grams "on the fly".
# Thus, when calling the `fit` we will pass an extra argument `n_grams=3` to indicate
# we want SS3 to learn to recognize important words, bigrams, and 3-grams [*].
# Additionally, we will name our model "movie_review_3grams" so that we can save it
# and load it later from the `PySS3 Command Line` to perform the hyper-parameter
# optimization to find better hyper-parameters values.
#
# [*] If you're curious and want to know how this is actually done by SS3, read the
#     paper "t-SS3: a text classifier with dynamic n-grams for early risk detection
#     over text streams" (preprint available here: https://arxiv.org/abs/1911.06147.
clf = SS3(name="movie_review_3grams")
clf.fit(x_train, y_train, n_grams=3)  # <-- note the n_grams=3 argument here

# As mentioned above, we will save this trained model for later use.
clf.save_model()

# Now let's see if the performance has improved...
y_pred = clf.predict(x_test)
print("Accuracy:", accuracy_score(y_pred, y_test))

# Yeah, the accuracy slightly improved but more importantly, we should now see that
# the model has learned "more intelligent patterns" involving sequences of words when
# using the interactive "live test" to observe what our model has learned (like
# "was supposed to", "has nothing to", "low budget", "your money", etc. for the
# "negative" class). Let's see...
Live_Test.run(clf, x_test, y_test)

# As described in the "Hyper-parameter Optimization" section of the tutorial,
# (https://pyss3.readthedocs.io/en/latest/tutorials/movie-review.html#hyper-parameter-optimization)
# after performing hyper-parameters optimization using
# the `PySS3 Command Line`, we found out that, for example, the following
# hyper-parameter values will slightly improve our classification performance
clf.set_hyperparameters(s=.44, l=.48, p=1.1)


# Let's see if it's true...
y_pred = clf.predict(x_test)
print("Accuracy:", accuracy_score(y_pred, y_test))

# Great! accuracy improved. Fortunately, this time we got lucky and the default
# hyper-parameters were also quite good.
