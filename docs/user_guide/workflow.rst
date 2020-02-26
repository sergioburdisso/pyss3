.. _workflow:

************
The Workflow
************

PySS3 provides two main types of workflow: classic and "command line". Both workflows are briefly described below.

.. _classic-workflow:
Classic Workflow
================

As usual, importing the needed classes and functions from the package, the user writes a python script to train and test the classifiers.


.. _pyss3-workflow:
Command-Line Workflow
=====================

When you install the package (for instance by using ``pip install pyss3``) a new command (``pyss3``) is automatically added to your environment's command line. This command allows you to access to the _PySS3 Command Line_, an interactive command-line query tool. This workflow consist of using this tool to carry out the whole machine learning pipeline (model selection, training, testing, etc.), which provides a faster way to perform experimentations since the user doesn't have to write any python script. Plus, this Command Line tool allows the user to actively interact  "on the fly" with the models being developed.



Note: :ref:`tutorials` are presented in two versions, one for each workflow type, so that the reader can choose the workflow that best suit her/his needs.
