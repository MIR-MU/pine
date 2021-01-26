# -*- coding: utf-8 -*-

from __future__ import annotations

from logging import getLogger
from pathlib import Path
import pickle
from typing import Dict, Optional, Literal, Union

from .configuration import FASTTEXT_PARAMETERS, MODEL_BASENAMES, PICKLE_PROTOCOL
from .util import stringify_parameters
from .corpus import get_corpus, Corpus

from gensim.models import FastText, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import humanize


LOGGER = getLogger(__name__)


class LanguageModel:
    def __init__(self,
                 corpus_name: str,
                 model_dir: Union[Path, str],
                 cache_dir: Union[Path, str],
                 corpus_dir: Union[Path, str],
                 dataset_dir: Union[Path, str],
                 subwords: bool = True,
                 positions: Literal[False, 'full', 'constrained'] = 'constrained',
                 use_vocab_from: LanguageModel = None,
                 extra_fasttext_parameters: Optional[Dict] = None):

        self.corpus_name = corpus_name
        self._model_dir = Path(model_dir)
        self._cache_dir = Path(cache_dir)
        self.corpus_dir = Path(corpus_dir)
        self.dataset_dir = Path(dataset_dir)
        self.subwords = subwords
        self.positions = positions
        self.use_vocab_from = use_vocab_from
        self.extra_fasttext_parameters = extra_fasttext_parameters

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._vectors = None
        self._training_duration = None

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
        try:
            return self._load_training_duration()
        except IOError:
            self._train_model()
            return self._load_training_duration()

    def __repr__(self) -> str:
        return '<<{} {}>>'.format(self.__class__.__name__, self.basename)

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
    def basename(self) -> str:
        basename = []
        basename.append(self.corpus_name)
        basename.append(self.language)
        basename.append(MODEL_BASENAMES(self.subwords, self.positions))
        basename.append(stringify_parameters(self.extra_fasttext_parameters))
        basename = filter(len, basename)
        return '-'.join(basename)

    @property
    def model_dir(self) -> str:
        return self._model_dir / self.basename

    @property
    def cache_dir(self) -> str:
        return self._cache_dir / self.basename

    def print_model_files(self):
        for path in self.model_dir, self.cache_dir:
            for path in path.glob('**/*'):
                if not path.is_file():
                    continue
                size = path.stat().st_size
                size = humanize.naturalsize(size)
                print('{}\t{}'.format(size, path))

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

    def _load_training_duration(self) -> float:
        if self._training_duration is not None:
            return self._training_duration
        message = 'Loading training duration for {} from {}'
        message = message.format(self, self._training_duration_path)
        LOGGER.debug(message)
        with self._training_duration_path.open('rt') as f:
            self._training_duration = float(f)
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

        LOGGER.info('Saving model for {} to {}'.format(self, self._model_path))
        self._model.save(str(self._model_path))
        LOGGER.debug('Saving vectors for {} to {}'.format(self, self._vectors_path))
        self._vectors.save_word2vec_format(str(self._vectors_path))
        message = 'Saving training duration for {} to {}'
        message = message.format(self, self._training_duration_path)
        LOGGER.debug(message)
        with self._training_duration_path.open('wt') as f:
            print(self._training_duration, file=f)


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
