# -*- coding: utf-8 -*-

from .text_classification import (
    evaluate as evaluate_text_classification,
    get_dataset_paths as get_text_classification_datasets,
)
from .word_analogy import (
    evaluate as evaluate_word_analogy,
    get_dataset_path as get_word_analogy_dataset,
)

__all__ = [
    'evaluate_text_classification',
    'get_text_classification_datasets',
    'evaluate_word_analogy',
    'get_word_analogy_dataset',
]
