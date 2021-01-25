# -*- coding: utf-8 -*-

from .text_classification import (
    evaluate as evaluate_text_classification,
    get_dataset_paths as get_text_classification_datasets,
)
from .word_analogy import (
    evaluate as evaluate_word_analogy,
    get_dataset_path as get_word_analogy_dataset,
    WordAnalogyResult,
)
from .language_modeling import (
    evaluate as evaluate_language_modeling,
    get_dataset_paths as get_language_modeling_dataset,
    LanguageModelingResult,
)

__all__ = [
    'get_text_classification_datasets',
    'evaluate_text_classification',
    'get_word_analogy_dataset',
    'evaluate_word_analogy',
    'get_language_modeling_dataset',
    'WordAnalogyResult',
    'evaluate_language_modeling',
    'LanguageModelingResult',
]
