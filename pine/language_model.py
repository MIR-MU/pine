# -*- coding: utf-8 -*-

from __future__ import annotations

from logging import getLogger
from pathlib import Path
import pickle
from typing import Dict, Optional, Literal

from .configuration import FASTTEXT_PARAMETERS, MODEL_BASENAMES, PICKLE_PROTOCOL
from .util import stringify_parameters
from .corpus import get_corpus, Corpus

from gensim.models import FastText, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec


LOGGER = getLogger(__name__)


class LanguageModel:
    def __init__(self, corpus_name: str,
                 model_dir: Path, corpus_dir: Path, dataset_dir: Path, cache_dir: Path,
                 subwords: bool = True,
                 positions: Literal[False, 'full', 'constrained'] = 'constrained',
                 use_vocab_from: LanguageModel = None,
                 extra_fasttext_parameters: Optional[Dict] = None):
        self.corpus_name = corpus_name
        self.model_dir = model_dir
        self.corpus_dir = corpus_dir
        self.dataset_dir = dataset_dir
        self.cache_dir = cache_dir
        self.subwords = subwords
        self.positions = positions
        self.use_vocab_from = use_vocab_from
        self.extra_fasttext_parameters = extra_fasttext_parameters

        self._model = None
        self._vectors = None

    @property
    def model(self) -> FastText:
        try:
            return self._load_model()
        except IOError:
            self._train_model()
            return self._load_model()

    @property
    def vectors(self) -> KeyedVectors:
        try:
            return self._load_vectors()
        except IOError:
            self._train_model()
            return self._load_vectors()

    @property
    def training_duration(self) -> float:
        for callback in self.model.callbacks:
            if isinstance(callback, TrainingDurationMeasure):
                return callback.total_seconds
        raise ValueError('Training duration not found in model {}'.format(self._model_path))

    def __repr__(self) -> str:
        return '<<{} {}>>'.format(self.__class__.__name__, self.basename.with_suffix('.*'))

    @property
    def corpus(self) -> Corpus:
        return get_corpus(self.corpus_name, self.corpus_dir, self.language)

    @property
    def fasttext_parameters(self) -> Dict:
        return {
            **FASTTEXT_PARAMETERS['baseline'],
            **FASTTEXT_PARAMETERS['subwords'][self.subwords],
            **FASTTEXT_PARAMETERS['positions'][self.positions],
            **(self.extra_fasttext_parameters or dict()),
        }

    @property
    def basename(self) -> Path:
        basename = []
        basename.append(self.corpus_name)
        basename.append(self.language)
        basename.append(MODEL_BASENAMES(self.subwords, self.positions))
        basename.append(stringify_parameters(self.extra_fasttext_parameters))
        basename = filter(len, basename)
        return self.model_dir / '-'.join(basename)

    @property
    def _bare_model_path(self) -> Path:
        return self.basename.with_suffix('.bare-model')

    @property
    def _model_path(self) -> Path:
        return self.basename.with_suffix('.model')

    @property
    def _vectors_path(self) -> Path:
        return self.basename.with_suffix('.vec')

    def _load_model(self) -> FastText:
        if self._model is not None:
            return self._model
        LOGGER.debug('Loading model for {} from {}'.format(self, self._model_path))
        self._model = FastText.load(str(self._model_path), mmap='r')
        self._vectors = self._model.wv
        return self._model

    def _load_vectors(self) -> KeyedVectors:
        if self._vectors is not None:
            return self._vectors
        LOGGER.debug('Loading vectors for {} from {}'.format(self, self._vectors_path))
        self._vectors = KeyedVectors.load_word2vec_format(str(self._vectors_path))
        return self._vectors

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
        if not self.subwords:
            model.wv.vectors = model.wv.vectors_vocab  # Apply fix from Gensim issue #2891

        self._model = model
        self._vectors = model.wv

        LOGGER.info('Saving model for {} to {}'.format(self, self._model_path))
        model.save(str(self._model_path))
        LOGGER.debug('Saving vectors for {} to {}'.format(self, self._vectors_path))
        model.wv.save_word2vec_format(str(self._vectors_path))


class TrainingDurationMeasure(CallbackAny2Vec):
    def __init__(self):
        self.start_time = None
        self.total_seconds = 0.0

    def on_epoch_begin(self, model):
        from datetime import datetime
        self.start_time = datetime.now()

    def on_epoch_end(self, model):
        from datetime import datetime
        finish_time = datetime.now()
        self.total_seconds += (finish_time - self.start_time).total_seconds()
