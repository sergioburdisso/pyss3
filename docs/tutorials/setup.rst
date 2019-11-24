.. _tutorial-setup:

Tutorial Setup
==============

To get started with the tutorials, we need to download the content of the `examples <https://github.com/sergioburdisso/pyss3/tree/master/examples>`__ folder from the `PySS3 repo <https://github.com/sergioburdisso/pyss3>`__. This folder contains the dataset as well as other files needed for the tutorials. The faster way to do this is to download the entire repository either by:

- cloning it:

.. code:: console

    git clone https://github.com/sergioburdisso/pyss3

- or `downloading a zipped version <https://github.com/sergioburdisso/pyss3/archive/master.zip>`__.


Additionally, we strongly recommend creating a new `conda enviroment <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html>`__ to install everything we need for the tutorials, including PySS3. We will call this new environemnt "pyss3tutos":

.. code:: console

    conda create --name pyss3tutos python=3

.. note:: don't have `conda <https://docs.conda.io/projects/conda/en/latest/>`__ in your system? don't worry, read its `installation guide <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`__! :)

Now, activate our new conda environemnt:

.. code:: console

    conda activate pyss3tutos

Before installing PySS3, if you want to run the tutorials using the `Jupyter Notebook <https://jupyter.org/install>`__, install ``ipykernel`` to add our new environment to the Jupyter Notebook kernels:

.. code:: console

    conda install ipykernel

Finally! install PySS3:

.. code:: console

    pip install pyss3

.. seealso:: If you want to install pyss3 using another method, you should check out the :ref:`installation` page.