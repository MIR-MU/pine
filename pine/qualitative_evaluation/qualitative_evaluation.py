# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import deque
from logging import getLogger
from itertools import chain
from typing import Sequence, Optional, Iterable, List, Dict, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.cluster import AgglomerativeClustering

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from ..configuration import (
    NUM_PRINTED_TOP_WORDS,
    NUM_PRINTED_BOTTOM_WORDS,
    FEATURE_CLUSTERING_PARAMETERS,
    NUM_FEATURE_CLUSTERS,
    WORD_KINDS,
)
from ..language_model import LanguageModel


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
    sorted_indices = inner_products.argsort()[::-1]
    log_message_format = '{}[{{}}]{}'.format(
        ' '.join((*left_context, '') if left_context else left_context),
        ' '.join(('', *right_context) if right_context else right_context),
    )
    return _yield_ranked_words(language_model, sorted_indices, log_message_format)


def classify_words(language_model: LanguageModel, kind: str) -> Dict[str, str]:
    if kind not in WORD_KINDS:
        message = 'Unknown kind {} (known kinds: {})'
        message = message.format(kind, ', '.join(WORD_KINDS))
        raise ValueError(message)

    if kind == 'context':
        word_vectors = language_model.input_vectors
    elif kind == 'masked':
        word_vectors = language_model.output_vectors
    else:
        raise ValueError('Vectors for kind {} not yet implemented'.format(kind))

    cluster_labels, clusters = zip(*iter(language_model.positional_feature_clusters))
    word_importances = np.vstack([
        np.abs(word_vectors.T[cluster].T).sum(axis=1) * 1.0 / len(cluster)
        for cluster
        in clusters
    ])
    predicted_cluster_indices = np.argmax(word_importances, axis=0)
    predicted_clusters = dict()
    predicted_cluster_counts = dict()
    for word, predicted_cluster_index in zip(language_model.words, predicted_cluster_indices):
        cluster_label = cluster_labels[predicted_cluster_index]
        predicted_clusters[word] = cluster_label
        if cluster_label not in predicted_cluster_counts:
            predicted_cluster_counts[cluster_label] = 0
        predicted_cluster_counts[cluster_label] += 1

    message = ', '.join(
        '{:.2f}% {}'.format(count * 100.0 / len(language_model.words), cluster_label)
        for cluster_label, count
        in sorted(predicted_cluster_counts.items(), key=lambda x: x[1], reverse=True)
    )
    message = 'Predicted word classes: {}'.format(message)
    LOGGER.info(message)

    return predicted_clusters


def get_position_importance(language_model: LanguageModel) -> PositionImportance:
    if not language_model.positions:
        raise ValueError('{} is not a positional model'.format(language_model))
    importance = np.linalg.norm(language_model.positional_vectors, axis=1)
    max_importance = importance.max()
    importance = importance / max_importance if max_importance > 0 else importance
    return PositionImportance(language_model, importance)


class PositionImportance:
    def __init__(self, language_model: LanguageModel, data: np.ndarray):
        self.language_model = language_model
        self.data = data

    def __iter__(self) -> Iterable[float]:
        return iter(self.data)

    def plot(self) -> 'Figure':
        from .view import plot_position_importance
        return plot_position_importance(self.language_model)

    def __repr__(self) -> str:
        return ' position importance of {}'.format(self.language_model)

    def _repr_html_(self) -> str:
        figure = self.plot()
        return figure._repr_html_()


def cluster_positional_features(language_model: LanguageModel) -> ClusteredPositionalFeatures:
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
    return ClusteredPositionalFeatures(language_model, clusters)


class ClusteredPositionalFeatures:
    def __init__(self, language_model: LanguageModel, data: Dict[str, List[int]]):
        self.language_model = language_model
        self.data = data

    def __iter__(self) -> Iterable[Tuple[str, List[int]]]:
        return iter(self.data.items())

    def __contains__(self, label: str) -> bool:
        return label in self.data

    def __getitem__(self, label: str) -> List[int]:
        return self.data[label]

    def plot(self) -> 'Figure':
        from .view import plot_clustered_positional_features
        return plot_clustered_positional_features(self.language_model)

    def __repr__(self) -> str:
        return 'Clustered positional features of {}'.format(self.language_model)

    def _repr_html_(self) -> str:
        figure = self.plot()
        return figure._repr_html_()


def _yield_ranked_words(language_model: LanguageModel, sorted_indices: Iterable[int],
                        log_message_format: str = '{}') -> Iterable[str]:
    printed_top_words, printed_bottom_words = [], deque([], NUM_PRINTED_BOTTOM_WORDS)
    num_words = 0
    for index in sorted_indices:
        word = language_model.words[index]
        num_words += 1
        if num_words <= NUM_PRINTED_TOP_WORDS:
            printed_top_words.append(word)
        printed_bottom_words.append(word)
        yield word
    if num_words <= NUM_PRINTED_TOP_WORDS:
        message = log_message_format.format(', '.join(printed_top_words))
    elif num_words <= NUM_PRINTED_BOTTOM_WORDS:
        message = log_message_format.format(', '.join(printed_bottom_words))
    elif num_words <= NUM_PRINTED_TOP_WORDS + NUM_PRINTED_BOTTOM_WORDS:
        remaining_words_below_top = num_words - NUM_PRINTED_TOP_WORDS
        printed_words = [*printed_top_words, *printed_bottom_words[-remaining_words_below_top:]]
        message = log_message_format.format(', '.join(printed_words))
    else:
        printed_words = [*printed_top_words, '...', *printed_bottom_words]
        message = log_message_format.format(', '.join(printed_words))
    LOGGER.info(message)
