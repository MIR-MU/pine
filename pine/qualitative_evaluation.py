# -*- coding: utf-8 -*-

from __future__ import annotations

from logging import getLogger
from itertools import chain
from typing import Sequence, Optional, Iterable, List, Dict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from .configuration import NUM_PRINTED_TOP_WORDS, FEATURE_CLUSTERING_PARAMETERS, NUM_FEATURE_CLUSTERS
from .language_model import LanguageModel


LOGGER = getLogger(__name__)


def predict_masked_words(language_model: LanguageModel, sentence: Sequence[Optional[str]]) -> Iterable[str]:
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


def get_importance_of_positions(language_model: LanguageModel) -> np.ndarray:
    if not language_model.positions:
        raise ValueError('{} is not a positional model'.format(language_model))
    return np.linalg.norm(language_model.positional_vectors, axis=1)


def cluster_positional_features(language_model: LanguageModel) -> Dict[str, List[int]]:
    if not language_model.positions:
        raise ValueError('{} is not a positional model'.format(language_model))

    absolute_positional_vectors = np.abs(language_model.positional_vectors)
    window_center = len(absolute_positional_vectors) // 2
    left_context_features = absolute_positional_vectors[:window_center][::-1].T
    right_context_features = absolute_positional_vectors[window_center:].T
    context_difference = right_context_features - left_context_features

    num_feature_clusters = NUM_FEATURE_CLUSTERS[language_model.positions]
    clustering = AgglomerativeClustering(n_clusters=num_feature_clusters, **FEATURE_CLUSTERING_PARAMETERS)
    labels = clustering.fit(context_difference).labels_

    if language_model.positions == 'full':
        means = np.array([
            np.mean(absolute_positional_vectors.T[labels == label])
            for label in range(num_feature_clusters)
        ])
        informational_cluster_label = np.argmax(means)
    else:
        informational_cluster_label = None

    clusters = dict()
    num_antepositional, num_postpositional = 0, 0
    for index, label in enumerate(labels):
        if label == informational_cluster_label:
            label = 'informational'
        else:
            if np.mean(context_difference[labels == label]) < 0:
                if num_antepositional == 0:
                    label = 'antepositional'
                else:
                    label = 'antepositional #{}'.format(num_antepositional)
            else:
                if num_postpositional == 0:
                    label = 'postpositional'
                else:
                    label = 'postpositional #{}'.format(num_postpositional)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(index)
    return clusters
