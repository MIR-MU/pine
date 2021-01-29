# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import deque
import json
from logging import getLogger
from itertools import chain, product
from typing import Sequence, Optional, Iterable, List, Dict, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from ..configuration import (
    NUM_PRINTED_TOP_WORDS,
    NUM_PRINTED_BOTTOM_WORDS,
    FEATURE_CLUSTERING_PARAMETERS,
    NUM_FEATURE_CLUSTERS,
    WORD_KINDS,
    EXAMPLE_SENTENCES,
    JSON_DUMP_PARAMETERS,
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
    message = 'Word classes: {}'.format(message)
    LOGGER.info(message)

    return predicted_clusters


def produce_example_sentences(language_model: LanguageModel, cluster_label: str) -> ExampleSentences:
    result_path = language_model.model_dir / 'example_sentences'
    result_path.mkdir(exist_ok=True)
    result_path = result_path / cluster_label
    result_path = result_path.with_suffix('.json')
    try:
        with result_path.open('rt') as rf:
            result = json.load(rf)
    except IOError:
        cluster = language_model.positional_feature_clusters[cluster_label]
        positional_vectors = language_model.positional_vectors.T[cluster].T
        restrict_vocab = EXAMPLE_SENTENCES['restrict_vocab']
        whitelist = EXAMPLE_SENTENCES['whitelist'][language_model.language]
        blacklist = EXAMPLE_SENTENCES['blacklist'][language_model.language]

        def is_masked(word: str) -> bool:
            if whitelist is not None and word not in whitelist:
                return True
            if blacklist is not None and blacklist(word):
                return True
            if language_model.classify_context_word(word) != cluster_label:
                return True
            return False

        mask = np.array([
            [is_masked(word)] * len(cluster)
            for word
            in language_model.words[:restrict_vocab]
        ])
        input_vectors = language_model.input_vectors.T[cluster].T[:restrict_vocab]
        input_vectors = [
            np.ma.array(positional_vectors[position] * input_vectors, mask=mask)
            for position
            in range(len(positional_vectors))
        ]
        output_vectors = language_model.output_vectors.T[cluster].T[:restrict_vocab]
        output_vectors = np.ma.array(output_vectors, mask=mask)

        window_center = len(positional_vectors) // 2
        min_position, max_position = EXAMPLE_SENTENCES['restrict_positions']
        min_position = min_position + window_center - (1 if min_position > 0 else 0)
        max_position = max_position + window_center - (1 if max_position > 0 else 0)

        positions = range(len(positional_vectors))
        positions = filter(lambda x: x >= min_position and x <= max_position, positions)
        positions = list(positions)
        positions = product(positions, positions)
        positions = filter(lambda x: x[0] < x[1], positions)
        positions = list(positions)
        positions = tqdm(positions, desc='Trying different pairs of positions')

        best_context_word_index, best_masked_word_index = None, None
        best_first_position, best_second_position = None, None
        best_effect = 0.0
        for first_position, second_position in positions:
            first_scores = input_vectors[first_position].dot(output_vectors.T)
            second_scores = input_vectors[second_position].dot(output_vectors.T)
            effects = np.abs(_sigmoid(first_scores) - _sigmoid(second_scores))
            context_word_index, masked_word_index = np.unravel_index(effects.argmax(), effects.shape)
            first_score = first_scores[context_word_index, masked_word_index]
            second_score = second_scores[context_word_index, masked_word_index]
            effect = effects[context_word_index, masked_word_index]
            if effect > best_effect:
                best_context_word_index, best_masked_word_index = context_word_index, masked_word_index
                best_first_position, best_second_position = first_position, second_position
                best_first_score, best_second_score = first_score, second_score
                best_effect = effect

        best_context_word = language_model.words[best_context_word_index]
        best_masked_word = language_model.words[best_masked_word_index]
        result = (
            window_center,
            best_context_word,
            best_masked_word,
            best_first_position,
            float(best_first_score),
            best_second_position,
            float(best_second_score),
            float(best_effect),
        )
        with result_path.open('wt') as wf:
            json.dump(result, wf, **JSON_DUMP_PARAMETERS)

    return ExampleSentences(cluster_label, *result)


class ExampleSentences:
    def __init__(self, cluster_label: str, window_center: int, context_word: str,
                 masked_word: str, first_position: int, first_score: float,
                 second_position: int, second_score: float, effect: float):
        self.cluster_label = cluster_label
        self.window_center = window_center
        self.context_word = context_word
        self.masked_word = masked_word
        self.first_position = first_position
        self.first_score = first_score
        self.second_position = second_position
        self.second_score = second_score
        self.effect = effect

    def __repr__(self) -> str:
        first_position, second_position = self.first_position, self.second_position
        if self.first_position >= self.window_center:
            first_position += 1
        if self.second_position >= self.window_center:
            second_position += 1
        positions = (first_position, second_position)
        positions = range(min(self.window_center, *positions), max(self.window_center, *positions) + 1)

        first_sentence, second_sentence = [], []
        for position in positions:
            if position == self.window_center:
                masked_word = '[{}]'.format(self.masked_word)
                first_sentence.append(masked_word)
                second_sentence.append(masked_word)
            else:
                first_sentence.append(self.context_word if position == first_position else '?')
                second_sentence.append(self.context_word if position == second_position else '?')

        scores = [self.first_score, self.second_score]
        sentences = [first_sentence, second_sentence]
        message = 'Example sentences with the largest difference in probability ({:.2f}%) for cluster {}:'
        message = [message.format(self.effect * 100.0, self.cluster_label)]
        for score, sentence in sorted(zip(scores, sentences), reverse=False):
            sentence = ' '.join(sentence)
            probability = 100.0 * _sigmoid(score)
            message.append('- {} (score {:.2f}, probability {:.2f}%)'.format(sentence, score, probability))
        message = '\n'.join(message)
        return message


def _sigmoid(value: np.array) -> np.array:
    return np.exp(-np.logaddexp(0, -value))


def get_position_importance(language_model: LanguageModel) -> PositionImportance:
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
