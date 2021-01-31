# -*- coding: utf-8 -*-

from __future__ import annotations

from datetime import datetime, timedelta
from logging import getLogger
from pathlib import Path
import pickle
from typing import Dict, Optional, Union, Sequence, Tuple, Iterable, TYPE_CHECKING

from .configuration import FASTTEXT_PARAMETERS, MODEL_BASENAMES, MODEL_FRIENDLY_NAMES, PICKLE_PROTOCOL
from .util import stringify_parameters
from .corpus import get_corpus, Corpus

if TYPE_CHECKING:
    from .qualitative_evaluation import (
        PositionImportance,
        ClusteredPositionalFeatures,
        ExampleSentences,
        Sentence,
        SentenceProbability,
    )
    from .quantitative_evaluation import WordAnalogyResult, LanguageModelingResult

from gensim.models import FastText, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import humanize
import numpy as np


LOGGER = getLogger(__name__)


class LanguageModel:
    def __init__(self,
                 corpus: str,
                 workspace: Union[Path, str] = '.',
                 language: str = 'en',
                 subwords: bool = True,
                 positions: Union[bool, str] = 'constrained',
                 use_vocab_from: LanguageModel = None,
                 friendly_name: str = None,
                 extra_fasttext_parameters: Optional[Dict] = None):

        self._corpus = corpus
        self.workspace = Path(workspace)
        self.language = language
        self.subwords = subwords
        self.positions = positions
        self.use_vocab_from = use_vocab_from
        self._friendly_name = friendly_name
        self.extra_fasttext_parameters = extra_fasttext_parameters

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._vectors = None
        self._training_duration = None
        self._position_importance = None
        self._positional_feature_clusters = None
        self._classified_context_words = None

    @property
    def model(self) -> FastText:
        try:
            return self._load_model()[0]
        except IOError:
            self._train_model()
            return self._load_model()[0]

    @property
    def vectors(self) -> KeyedVectors:
        try:
            return self._load_model()[1]
        except IOError:
            try:
                return self._load_vectors()
            except IOError:
                self._train_model()
                return self._load_vectors()

    @property
    def training_duration(self) -> float:
        try:
            return self._load_training_duration()
        except IOError:
            self._train_model()
            return self._load_training_duration()

    @property
    def input_vectors(self) -> np.ndarray:
        return self.vectors.vectors

    @property
    def positional_vectors(self) -> np.ndarray:
        if not self.positions:
            raise ValueError('{} is not a positional model'.format(self))
        return self.model.wv.vectors_positions

    @property
    def output_vectors(self) -> np.ndarray:
        if 'syn1' in vars(self.model):
            return self.model.syn1
        return self.model.syn1neg

    @property
    def position_importance(self) -> 'PositionImportance':
        if self._position_importance is None:
            from .qualitative_evaluation import get_position_importance
            self._position_importance = get_position_importance(self)
        return self._position_importance

    @property
    def positional_feature_clusters(self) -> 'ClusteredPositionalFeatures':
        if self._positional_feature_clusters is None:
            from .qualitative_evaluation import cluster_positional_features
            self._positional_feature_clusters = cluster_positional_features(self)
        return self._positional_feature_clusters

    def predict_masked_words(self, sentence: 'Sentence') -> Iterable[str]:
        from .qualitative_evaluation import predict_masked_words
        return predict_masked_words(self, sentence)

    def get_masked_word_probability(self, sentence: 'Sentence', masked_word: str,
                                    cluster_label: Optional[str] = None) -> 'SentenceProbability':
        from .qualitative_evaluation import get_masked_word_probability
        return get_masked_word_probability(self, sentence, masked_word, cluster_label)

    @property
    def classified_context_words(self) -> Dict[str, str]:
        if self._classified_context_words is None:
            from .qualitative_evaluation import classify_words
            self._classified_context_words = classify_words(self, 'context')
        return self._classified_context_words

    def classify_context_word(self, word: str) -> str:
        return self.classified_context_words[word]

    def produce_example_sentences(self, cluster_label: str) -> 'ExampleSentences':
        from .qualitative_evaluation import produce_example_sentences
        return produce_example_sentences(self, cluster_label)

    @property
    def words(self) -> Sequence[str]:
        return self.vectors.index_to_key

    @property
    def word_analogy(self) -> 'WordAnalogyResult':
        from .quantitative_evaluation import get_word_analogy_dataset, evaluate_word_analogy
        dataset = get_word_analogy_dataset(self.language, self.dataset_dir)
        return evaluate_word_analogy(dataset, self)

    @property
    def language_modeling(self) -> 'LanguageModelingResult':
        from .quantitative_evaluation import get_language_modeling_dataset, evaluate_language_modeling
        dataset = get_language_modeling_dataset(self.language, self.dataset_dir)
        return evaluate_language_modeling(dataset, self)

    def __str__(self) -> str:
        return self.basename

    def __repr__(self) -> str:
        training_duration = timedelta(seconds=self.training_duration)
        training_duration = humanize.naturaldelta(training_duration)
        model_files_size = humanize.naturalsize(sum(size for _, size in self.model_files))
        cache_files_size = humanize.naturalsize(sum(size for _, size in self.cache_files))
        lines = [
            'Language model: {}'.format(self.basename),
            'Disk size: {} (+ {} in cache)'.format(model_files_size, cache_files_size),
            'Training duration: {}'.format(training_duration),
        ]
        return '\n'.join(lines)

    def _files(self, dir_path: Path) -> Iterable[Tuple[Path, int]]:
        for path in dir_path.glob('**/*'):
            if not path.is_file():
                continue
            size = path.stat().st_size
            yield (path, size)

    @property
    def model_files(self) -> Iterable[Tuple[Path, int]]:
        return self._files(self.model_dir)

    @property
    def cache_files(self) -> Iterable[Tuple[Path, int]]:
        return self._files(self.cache_dir)

    def print_files(self):
        lines = []
        lines.extend(['Model files:', ''])
        for path, size in sorted(self.model_files, key=lambda x: x[1], reverse=True):
            size = humanize.naturalsize(size)
            lines.append('\t{:10}\t{}'.format(size, path))
        lines.extend(['', 'Cache files:', ''])
        for path, size in sorted(self.cache_files, key=lambda x: x[1], reverse=True):
            size = humanize.naturalsize(size)
            lines.append('\t{:10}\t{}'.format(size, path))
        print('\n'.join(lines))

    @property
    def corpus(self) -> Corpus:
        return get_corpus(self._corpus, self.corpus_dir, self.language)

    @property
    def fasttext_parameters(self) -> Dict:
        return {
            **FASTTEXT_PARAMETERS['baseline'],
            **FASTTEXT_PARAMETERS['subwords'][self.subwords],
            **FASTTEXT_PARAMETERS['positions'][self.positions],
            **(self.extra_fasttext_parameters or dict()),
        }

    @property
    def basename(self) -> str:
        basename = []
        basename.append(self._corpus)
        basename.append(self.language)
        basename.append(MODEL_BASENAMES(self.subwords, self.positions))
        if self.extra_fasttext_parameters is not None:
            basename.append(stringify_parameters(self.extra_fasttext_parameters))
        basename = filter(len, basename)
        return '-'.join(basename)

    @property
    def friendly_name(self) -> str:
        if self._friendly_name is not None:
            return self._friendly_name
        return MODEL_FRIENDLY_NAMES(self.subwords, self.positions)

    @property
    def model_dir(self) -> str:
        return self.workspace / 'model' / self.basename

    @property
    def cache_dir(self) -> str:
        return self.workspace / 'cache' / self.basename

    @property
    def corpus_dir(self) -> str:
        return self.workspace / 'corpus'

    @property
    def dataset_dir(self) -> str:
        return self.workspace / 'dataset'

    @property
    def _bare_model_path(self) -> Path:
        bare_model_path = self.model_dir / 'bare_model'
        return bare_model_path

    @property
    def _model_path(self) -> Path:
        model_path = self.model_dir / 'model'
        return model_path

    @property
    def _vectors_path(self) -> Path:
        vectors_path = self.model_dir / 'model'
        vectors_path = vectors_path.with_suffix('.vec')
        return vectors_path

    @property
    def _training_duration_path(self) -> Path:
        vectors_path = self.model_dir / 'training_duration'
        vectors_path = vectors_path.with_suffix('.txt')
        return vectors_path

    def _load_model(self) -> Tuple[FastText, KeyedVectors]:
        if self._model is not None:
            return (self._model, self._vectors)
        LOGGER.debug('Loading model from {}'.format(self._model_path))
        self._model = FastText.load(str(self._model_path), mmap='r')
        self._vectors = self._model.wv
        return (self._model, self._vectors)

    def _load_vectors(self) -> KeyedVectors:
        if self._vectors is not None:
            return self._vectors
        LOGGER.debug('Loading vectors from {}'.format(self._vectors_path))
        self._vectors = KeyedVectors.load_word2vec_format(str(self._vectors_path))
        return self._vectors

    def _load_training_duration(self) -> float:
        if self._training_duration is not None:
            return self._training_duration
        message = 'Loading training duration from {}'.format(self._training_duration_path)
        LOGGER.debug(message)
        with self._training_duration_path.open('rt') as f:
            self._training_duration = float(next(f))
        return self._training_duration

    def _build_vocab(self) -> FastText:
        try:
            with self._bare_model_path.open('rb') as rf:
                LOGGER.debug('Loading vocab for {} from {}'.format(self, self._bare_model_path))
                saved_values = pickle.load(rf)
        except IOError:
            if self.use_vocab_from is None:
                bare_model = FastText(**self.fasttext_parameters)
                build_vocab_parameters = {
                    **FASTTEXT_PARAMETERS['build_vocab'],
                    **{
                        key: value
                        for (key, value) in self.fasttext_parameters.items()
                        if key in FASTTEXT_PARAMETERS['build_vocab'].keys()
                    },
                }
                LOGGER.info('Building vocab for {}'.format(self))
                bare_model.build_vocab(corpus_iterable=self.corpus, **build_vocab_parameters)
            else:
                if self.use_vocab_from._model is None:
                    LOGGER.warn('Using vocab of uninitialized {}'.format(self.use_vocab_from))
                LOGGER.debug('Using vocab of {} for {}'.format(self.use_vocab_from, self))
                bare_model = self.use_vocab_from.model
                self.use_vocab_from = None  # Free the reference to allow garbage collection
            saved_values = {'model_values': {}, 'wv_values': {}}
            for key in FASTTEXT_PARAMETERS['build_vocab_keys']:
                if key in vars(bare_model):
                    saved_values['model_values'][key] = bare_model.__dict__[key]
                elif key in vars(bare_model.wv):
                    saved_values['wv_values'][key] = bare_model.wv.__dict__[key]
                else:
                    message = 'Key {} not found in FastText model or its keyed vectors'.format(key)
                    raise KeyError(message)
            del bare_model
            with self._bare_model_path.open('wb') as wf:
                LOGGER.debug('Saving vocab for {} to {}'.format(self, self._bare_model_path))
                pickle.dump(saved_values, wf, protocol=PICKLE_PROTOCOL)

        model = FastText(**self.fasttext_parameters)
        for key, value in saved_values['model_values'].items():
            model.__dict__[key] = value
        for key, value in saved_values['wv_values'].items():
            model.wv.__dict__[key] = value
        model.wv.norms = None
        model.prepare_weights()

        return model

    def _train_model(self) -> FastText:
        model = self._build_vocab()
        training_duration_measure = TrainingDurationMeasure()
        train_parameters = {
            **FASTTEXT_PARAMETERS['train'],
            **{
                'total_examples': model.corpus_count,
                'total_words': model.corpus_total_words,
                'callbacks': [training_duration_measure],
            },
            **{
                key: value
                for (key, value) in self.fasttext_parameters.items()
                if key in FASTTEXT_PARAMETERS['train_keys']
            }
        }
        LOGGER.info('Training {}'.format(self))
        model.train(corpus_iterable=self.corpus, **train_parameters)
        model.callbacks = ()
        if not self.subwords:
            model.wv.vectors = model.wv.vectors_vocab  # Apply fix from Gensim issue #2891

        self._model = model
        self._vectors = model.wv
        self._training_duration = training_duration_measure.total_seconds

        LOGGER.info('Saving model to {}'.format(self._model_path))
        self._model.save(str(self._model_path))
        LOGGER.debug('Saving vectors to {}'.format(self._vectors_path))
        self._vectors.save_word2vec_format(str(self._vectors_path))
        message = 'Saving training duration to {}'.format(self._training_duration_path)
        LOGGER.debug(message)
        with self._training_duration_path.open('wt') as f:
            print(self._training_duration, file=f)


class TrainingDurationMeasure(CallbackAny2Vec):
    def __init__(self):
        self.start_time = None
        self.total_seconds = 0.0

    def on_epoch_begin(self, model):
        self.start_time = datetime.now()

    def on_epoch_end(self, model):
        finish_time = datetime.now()
        self.total_seconds += (finish_time - self.start_time).total_seconds()
