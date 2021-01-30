# -*- coding: utf-8 -*-

from .text_classification import (
    evaluate as evaluate_text_classification,
    get_dataset_paths as get_text_classification_datasets,
    Result as TextClassificationResult,
)
from .word_analogy import (
    evaluate as evaluate_word_analogy,
    get_dataset_path as get_word_analogy_dataset,
    Result as WordAnalogyResult,
)
from .language_modeling import (
    evaluate as evaluate_language_modeling,
    get_dataset_paths as get_language_modeling_dataset,
    Result as LanguageModelingResult,
    plot_language_modeling_results,
)

__all__ = [
    'get_text_classification_datasets',
    'evaluate_text_classification',
    'TextClassificationResult',

    'get_word_analogy_dataset',
    'evaluate_word_analogy',
    'WordAnalogyResult',

    'get_language_modeling_dataset',
    'evaluate_language_modeling',
    'LanguageModelingResult',
    'plot_language_modeling_results',
]
