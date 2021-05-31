# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import deque
import json
from logging import getLogger
from itertools import chain, product
from typing import Sequence, Optional, Iterable, List, Dict, Tuple, TYPE_CHECKING

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import minmax_scale
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
from .util import index_to_position, position_to_index

LOGGER = getLogger(__name__)


def predict_masked_words(language_model: LanguageModel, sentence: Sentence,
                         cluster_label: Optional[str] = None) -> Iterable[str]:
    context = _sentence_to_context(language_model, sentence)
    left_context, right_context = context
    context_vector = _context_to_vector(language_model, context, cluster_label)
    inner_products = language_model.output_vectors.dot(context_vector)
    sorted_indices = inner_products.argsort()[::-1]
    log_message_format = '{}[MASKED: {{}}]{}'.format(
        ' '.join((*left_context, '') if left_context else left_context),
        ' '.join(('', *right_context) if right_context else right_context),
    )
    return _yield_ranked_words(language_model, sorted_indices, log_message_format)


Sentence = Sequence[str]


def get_masked_word_probability(language_model: LanguageModel, sentence: Sentence,
                                masked_word: str,
                                cluster_label: Optional[str] = None) -> SentenceProbability:
    context = _sentence_to_context(language_model, sentence)
    context_vector = _context_to_vector(language_model, context, cluster_label)
    output_vector = _masked_word_to_vector(language_model, masked_word, cluster_label)
    score = context_vector.T.dot(output_vector)
    return SentenceProbability(sentence, masked_word, score)


class SentenceProbability:
    """The probability of a sentence given a masked word.

    Parameters
    ----------
    sentence : Sentence
        A sentence.
    masked_word : str
        A masked word.
    probability : float
        The probability of the sentence given the masked word.

    Attributes
    ----------
    sentence : Sentence
        A sentence.
    masked_word : str
        A masked word.
    probability : float
        The probability of the sentence given the masked word.

    """
    def __init__(self, sentence: Sentence, masked_word: str, score: float):
        self.sentence = sentence
        self.masked_word = masked_word
        self.score = score

    def __repr__(self) -> str:
        probability, = 100.0 * _sigmoid(np.array(self.score))
        sentence = ' '.join(
            word if word != '[MASK]' else '[MASKED: {}]'.format(self.masked_word)
            for word
            in self.sentence
        )
        return '{} (score {:.2f}, probability {:.2f}%)'.format(sentence, self.score, probability)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0, -value))


def _sentence_to_context(language_model: LanguageModel, sentence: Sentence) -> Context:
    num_masked_words = 0
    masked_word_index = None
    for word_index, word in enumerate(sentence):
        if word == '[MASK]':
            masked_word_index = word_index
            num_masked_words += 1

    if num_masked_words != 1:
        raise ValueError('Expected one masked word, found {}'.format(num_masked_words))

    window = language_model.model.window
    left_context = sentence[max(0, masked_word_index-window):masked_word_index]
    right_context = sentence[masked_word_index+1:min(len(sentence), masked_word_index+1+window)]
    return (left_context, right_context)


LeftContext = Sequence[str]
RightContext = Sequence[str]
Context = Tuple[LeftContext, RightContext]


def _context_to_vector(language_model: LanguageModel, context: Context,
                       cluster_label: Optional[str] = None) -> np.ndarray:
    left_context, right_context = context
    words = chain(left_context, right_context)
    if language_model.positions:
        min_position = -len(left_context)
        max_position = len(right_context)
        positions = range(min_position, max_position + 1)
        positions = filter(lambda x: x != 0, positions)
        positions_and_words = zip(positions, words)
    else:
        positions_and_words = enumerate(words)
    input_vectors, positional_vectors = [], []
    for position, word in positions_and_words:
        if word == '[PAD]':
            continue
        input_vector = language_model.vectors[word]
        input_vectors.append(input_vector)
        positional_vector = np.ones(len(input_vector), dtype=input_vector.dtype)
        if language_model.positions:
            index = position_to_index(language_model, position)
            positional_dimensions = language_model.positional_vectors.shape[1]
            positional_vector[:positional_dimensions] = language_model.positional_vectors[index]
        positional_vectors.append(positional_vector)
    input_vectors, positional_vectors = np.array(input_vectors), np.array(positional_vectors)
    if cluster_label is not None:
        cluster = language_model.positional_feature_clusters[cluster_label]
        input_vectors = input_vectors.T[cluster].T
        positional_vectors = positional_vectors.T[cluster].T
    context_vector = np.mean(input_vectors * positional_vectors, axis=0)
    return context_vector


def _context_word_to_vector(language_model: LanguageModel, context_word: str,
                            cluster_label: Optional[str] = None) -> np.ndarray:
    input_vector = language_model.vectors[context_word]
    if cluster_label is not None:
        cluster = language_model.positional_feature_clusters[cluster_label]
        input_vector = input_vector.T[cluster].T
    return input_vector


def _masked_word_to_vector(language_model: LanguageModel, masked_word: str,
                           cluster_label: Optional[str] = None) -> np.ndarray:
    masked_word_index = language_model.vectors.get_index(masked_word)
    output_vector = language_model.output_vectors[masked_word_index]
    if cluster_label is not None:
        cluster = language_model.positional_feature_clusters[cluster_label]
        output_vector = output_vector.T[cluster].T
    return output_vector


def _position_index_to_vector(language_model: LanguageModel, index: int,
                              cluster_label: Optional[str] = None) -> np.ndarray:
    positional_vector = language_model.positional_vectors[index]
    if cluster_label is not None:
        cluster = language_model.positional_feature_clusters[cluster_label]
        positional_vector = positional_vector.T[cluster].T
    return positional_vector


def _position_to_vector(language_model: LanguageModel, position: int,
                        cluster_label: Optional[str] = None) -> np.ndarray:
    index = position_to_index(language_model, position)
    positional_vector = _position_index_to_vector(language_model, index, cluster_label)
    return positional_vector


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

        min_position, max_position = EXAMPLE_SENTENCES['restrict_positions']
        min_position, max_position = int(min_position), int(max_position)
        min_position = position_to_index(language_model, min_position)
        max_position = position_to_index(language_model, max_position)

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
            effect = effects[context_word_index, masked_word_index]
            if effect > best_effect:
                best_context_word_index, best_masked_word_index = context_word_index, masked_word_index
                best_first_position, best_second_position = first_position, second_position
                best_effect = effect

        best_context_word = language_model.words[best_context_word_index]
        best_masked_word = language_model.words[best_masked_word_index]
        best_first_position = index_to_position(language_model, best_first_position)
        best_second_position = index_to_position(language_model, best_second_position)
        best_first_sentence, best_second_sentence = [], []
        for position in range(min(0, best_first_position), max(0, best_second_position) + 1):
            if position == 0:
                best_first_sentence.append('[MASK]')
                best_second_sentence.append('[MASK]')
            else:
                if position == best_first_position:
                    best_first_sentence.append(best_context_word)
                else:
                    best_first_sentence.append('[PAD]')
                if position == best_second_position:
                    best_second_sentence.append(best_context_word)
                else:
                    best_second_sentence.append('[PAD]')

        result = (
            best_masked_word,
            best_first_sentence,
            best_second_sentence,
        )
        with result_path.open('wt') as wf:
            json.dump(result, wf, **JSON_DUMP_PARAMETERS)

    return ExampleSentences(language_model, cluster_label, *result)


class ExampleSentences:
    """Two example sentences that characterize a cluster of positional features in a log-bilinear language model.

    A context word from a cluster of positional features will be placed on
    two different positions of a sentence, where it produces the greatest
    difference in masked word predictions. This is a useful illustration of
    the behavior and the purpose of a cluster of positional features.

    Parameters
    ----------
    language_model : :class:`~pine.language_model.LanguageModel`
        A log-bilinear language model.
    cluster_label : str
        A label of a cluster of positional features.
    masked_word : str
        A masked word.
    first_sentence : Sentence
        A sentence.
    second_sentence : Sentence
        Another sentence.

    Attributes
    ----------
    language_model : :class:`~pine.language_model.LanguageModel`
        A log-bilinear language model.
    cluster_label : str
        A label of a cluster of positional features.
    masked_word : str
        A masked word.
    first_sentence : Sentence
        A sentence.
    second_sentence : Sentence
        Another sentence.

    """
    def __init__(self, language_model: LanguageModel, cluster_label: str,
                 masked_word: str, first_sentence: Sentence,
                 second_sentence: Sentence):
        self.language_model = language_model
        self.cluster_label = cluster_label
        self.masked_word = masked_word
        self.first_sentence = first_sentence
        self.second_sentence = second_sentence

    def __repr__(self) -> str:
        first_result = get_masked_word_probability(self.language_model, self.first_sentence,
                                                   self.masked_word, self.cluster_label)
        second_result = get_masked_word_probability(self.language_model, self.second_sentence,
                                                    self.masked_word, self.cluster_label)
        results = [first_result, second_result]
        message = 'Example sentences for model {} using {} features:'
        message = message.format(self.language_model, self.cluster_label)
        message = [message, *map(repr, sorted(results, key=lambda x: x.score, reverse=True))]
        message = '\n'.join(message)
        return message


def get_position_importance(language_model: LanguageModel) -> PositionImportance:
    importance = np.linalg.norm(language_model.positional_vectors, axis=1)
    importance = minmax_scale(importance)
    return PositionImportance(language_model, importance)


class PositionImportance:
    """The importance of positions in a log-bilinear language model.

    Parameters
    ----------
    language_model : :class:`~pine.language_model.LanguageModel`
        The log-bilinear language model.
    data : np.ndarray
        The importance of positions.

    Attributes
    ----------
    language_model : :class:`~pine.language_model.LanguageModel`
        The log-bilinear language model.
    data : np.ndarray
        The importance of positions.

    """
    def __init__(self, language_model: LanguageModel, data: np.ndarray):
        self.language_model = language_model
        self.data = data

    def __iter__(self) -> Iterable[float]:
        return iter(self.data)

    def plot(self) -> 'Figure':
        from .view import plot_position_importance
        return plot_position_importance(self.language_model)

    def __repr__(self) -> str:
        return 'Position importance of {}'.format(self.language_model)

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
    """The clusters of positional features in a log-bilinear language model.

    Parameters
    ----------
    language_model : :class:`~pine.language_model.LanguageModel`
        A log-bilinear language model.
    data : dict of (str, list of int)
        Clusters of positional features in the log-bilinear language model.

    Attributes
    ----------
    language_model : :class:`~pine.language_model.LanguageModel`
        A log-bilinear language model.
    data : dict of (str, list of int)
        Clusters of positional features in the log-bilinear language model.

    """
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
