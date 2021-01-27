# -*- coding: utf-8 -*-

from __future__ import annotations

from logging import getLogger
from itertools import chain
from typing import List, Optional

import numpy as np

from .configuration import NUM_PRINTED_TOP_WORDS
from .language_model import LanguageModel


LOGGER = getLogger(__name__)


def predict_masked_words(language_model: LanguageModel, sentence: List[Optional[str]]) -> List[str]:
    num_masked_words = 0
    masked_word_index = None
    for word_index, word in enumerate(sentence):
        if word is None:
            masked_word_index = word_index
            num_masked_words += 1

    if num_masked_words != 1:
        raise ValueError('Expected one masked word, found {}'.format(num_masked_words))

    window = language_model.model.window
    left_context = sentence[max(0, masked_word_index-window):masked_word_index]
    right_context = sentence[masked_word_index+1:min(len(sentence), masked_word_index+1+window)]

    input_vectors, positional_vectors = [], []
    for position, word in enumerate(chain(left_context, right_context)):
        input_vector = language_model.vectors[word]
        input_vectors.append(input_vector)
        positional_vector = np.ones(len(input_vector), dtype=input_vector.dtype)
        if language_model.positions:
            positional_vector[:language_model.positional_vectors.shape[1]] = \
                language_model.positional_vectors[position]
        positional_vectors.append(positional_vector)
    input_vectors = np.array(input_vectors)
    positional_vectors = np.array(positional_vectors)
    context_vector = np.mean(input_vectors * positional_vectors, axis=0)

    inner_products = language_model.output_vectors.dot(context_vector)
    top_indices = inner_products.argsort()[::-1]
    printed_top_words, num_printed_top_words = [], min(NUM_PRINTED_TOP_WORDS, len(top_indices))
    for index in top_indices:
        top_word = language_model.words[index]
        if len(printed_top_words) <= num_printed_top_words:
            printed_top_words.append(top_word)
        if len(printed_top_words) == num_printed_top_words:
            message = '{}[{}]{}'.format(
                ' '.join((*left_context, '') if left_context else left_context),
                ', '.join((*printed_top_words, '...')),
                ' '.join(('', *right_context) if right_context else right_context),
            )
            LOGGER.info(message)
        yield top_word


def importance_of_positions(language_model: LanguageModel) -> np.ndarray:
    if not language_model.positions:
        raise ValueError('{} is not a positional model'.format(language_model))
    return np.linalg.norm(language_model.positional_vectors, axis=1)
