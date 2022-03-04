===============================
Position-Independent Embeddings
===============================

.. image:: https://github.com/MIR-MU/pine/workflows/Test/badge.svg
        :target: https://github.com/MIR-MU/pine/actions?query=workflow%3ATest
        :alt: Continuous Integration Status

.. image:: https://readthedocs.org/projects/position-independent-embeddings/badge/?version=latest
        :target: https://readthedocs.org/projects/position-independent-embeddings/?badge=latest
        :alt: Documentation Status

.. image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/MIR-MU/pine/blob/master/notebooks/tutorial.ipynb
        :alt: Open in Colab

Position-independent word embeddings (PInE) are word embeddings produced by
shallow log-bilinear language models (e.g. word2vec, fastText, or GLoVe) using
positional weighting. Positional weighting allows the models to distinguish
between words on different positions in a sentence and to produce better
position-independent representations of words. See our paper for details:

* Novotný, V., Štefánik, M., Ayetiran, E. F., Sojka, P. & Řehůřek, R. (2022).
  When FastText Pays Attention: Efficient Estimation of Word Representations
  using Constrained Positional Weighting. JUCS – Journal of Universal Computer
  Science (28, Issue 2, 181–201). https://doi.org/10.3897/jucs.69619

This Python package allows you to train, use, and evaluate position-independent
word embeddings.

.. image:: https://github.com/MIR-MU/pine/raw/main/images/pine.png

.. contents:: Table of Contents

Introduction
------------
Recent deep neural language models based on the Transformer architecture are
Turing-complete universal approximators that can understand language better
than humans on a number of natural language processing tasks.

In contrast, log-bilinear language models such as word2vec, fastText, and GLoVE
are shallow and use a simplifying bag-of-words representation of text, which
severely limits their predictive ability. However, they are fast and cheap to
train on large corpora and their internal *word embeddings* can be used for
transfer-learning to improve the performance of other models.

Our constrained positional model improves the bag-of-words representation of
text by allowing the model to react to the position of words in a sentence and
produce *position-independent word embeddings* without sacrificing the
simplicity and speed that is pivotal to the success of log-bilinear language
models. Unlike the positional model of `Mikolov et al. (2018)
<https://www.aclweb.org/anthology/L18-1008.pdf>`_, our model *constrains* the
capacity dedicated to modeling the positions of words, which improves the speed
of the model as well as its accuracy on a number of natural language processing
tasks.

Tutorials
---------
You can start from our Colab tutorial. In this tutorial, we are going to
produce our *position-independent word embeddings* and compare them with the
word embeddings of the subword model (fastText) of `Bojanowski et al.
(2017) <https://www.aclweb.org/anthology/Q17-1010.pdf>`_ and the positional
model of `Mikolov et al. (2018)
<https://www.aclweb.org/anthology/L18-1008.pdf>`_ on a number of natural
language processing tasks. We will also visualize the embeddings of positions,
which are a byproduct of the position-independent word embeddings, discuss
their properties and their possible applications for transfer learning.


.. |colab| image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/MIR-MU/pine/blob/master/notebooks/tutorial.ipynb
        :alt: Open in Colab

+---------------------------------------------------------------------------------+---------+
| Name                                                                            | Link    |
+=================================================================================+=========+
| Training + Masked Word Prediction + Language Modeling + Importance of Positions | |colab| |
+---------------------------------------------------------------------------------+---------+

Installation
------------

At the command line::

    $ pip install git+https://github.com/MIR-MU/pine.git

Or, if you have virtualenvwrapper installed::

    $ mkvirtualenv -p `which python3` pine
    (pine) $ pip install git+https://github.com/MIR-MU/pine.git

Development Team
----------------

* `Vít Novotný`_ <witiko@mail.muni.cz> Faculty of Informatics Masaryk University

.. _Vít Novotný: https://scholar.google.com/citations?user=XCkwOIoAAAAJ

Software Details
----------------

* Free software: LGPLv2.1 license
* Documentation: https://position-independent-embeddings.readthedocs.org.

Credits
-------

This package was created with Cookiecutter_ and the
`audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

Note
----

Remember that this is a research tool. 😉
