# coding: utf-8
"""
Movie Review (Sentiment Analysis) Tutorial Code.

This is the code for the tutorial "Movie Review - Classic Workflow"
(https://pyss3.readthedocs.io/en/latest/tutorials/movie-reviews.html#classic-workflow)
---
"""

# Before we begin, let's import needed modules!
from pyss3 import SS3
from pyss3.util import Dataset
from pyss3.server import Server

from sklearn.metrics import accuracy_score


# Ok, now we are ready to begin. Let's create a new SS3 instance.
# We will name it "movie_review" so that we can load the trained
# model from the `PySS3 Command Line` and perform a hyper-paremeter
# optimization to find the best hyper-parameters values.
clf = SS3(name="movie_review")


# What are the default hyper-paramter values? let's see
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


# Additionally, we will save this trained model for later use
# (from the `PySS3 Command Line` as the tutorial suggest)
clf.save_model()


# Now that the model has been trained, let's test it using the documents in `x_test`
y_pred = clf.predict(x_test)


# Let's see how good our model performed
print("Accuracy:", accuracy_score(y_pred, y_test))


# Not bad using the default hyper-parameters values,
# let's now manually analyze what our model has actually
# learned by using the interactive "live test"
Server.serve(clf, x_test, y_test)

# As described in the tutorial, after performing hyper-parameters
# optimization using the `PySS3 Command Line`, we found out that,
# for example, the following hyper-parameter values will slightly
# improve our classification performance
clf.set_hyperparameters(s=.44, l=.48, p=1.1)


# Let's see if it's true...
y_pred = clf.predict(x_test)
print("Accuracy:", accuracy_score(y_pred, y_test))


# Great! fortunately, we got lucky and the default hyper-parameters were quite good!
# What? Want to try this slightly better model? Ok, let's use the PySS3 server again :)
Server.serve(clf, x_test, y_test)
