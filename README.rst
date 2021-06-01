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
position-independent representations of words. See the papers for details:

* Mikolov, T., et al. “Advances in Pre-Training Distributed Word
  Representations.” *Proceedings of the Eleventh International Conference on
  Language Resources and Evaluation (LREC 2018).* 2018.
  http://www.lrec-conf.org/proceedings/lrec2018/pdf/721.pdf

* Novotný, V., et al. “When FastText Pays Attention: Efficient Estimation of
  Word Representations using Constrained Positional Weighting”. Manuscript
  submitted for publication.

This Python package allows you to train, use, and evaluate position-independent
word embeddings.

.. image:: https://github.com/MIR-MU/pine/raw/main/images/pine.png

* Free software: LGPLv2.1 license
* Documentation: https://position-independent-embeddings.readthedocs.org.
