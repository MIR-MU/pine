# -*- coding: utf-8 -*-

from .text_classification import (
    evaluate as evaluate_text_classification,
    get_dataset_paths as get_text_classification_datasets,
)

__all__ = ['evaluate_text_classification', 'get_text_classification_datasets']
