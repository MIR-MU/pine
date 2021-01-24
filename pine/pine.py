# -*- coding: utf-8 -*-

from logging import getLogger
from pathlib import Path
import pickle
from typing import Iterable, Dict, Optional
from ..config import FASTTEXT_PARAMETERS, PICKLE_PROTOCOL

from gensim.models import FastText, KeyedVectors


LOGGER = getLogger(__name__)


class LanguageModel(object):
    def __init__(self, corpus: Iterable[Iterable[str]], result_dir: Path,
                 name: str, subwords: bool = True, positions: bool = True,
                 gensim_kwargs: Optional[Dict] = None):
        self.corpus = corpus
        self.result_dir = result_dir
        self.name = name
        self.fasttext_parameters = {
            **FASTTEXT_PARAMETERS['baseline'],
            **FASTTEXT_PARAMETERS['subwords'][subwords],
            **FASTTEXT_PARAMETERS['positions'][positions],
            **gensim_kwargs,
        }
        self._model = None
        self._vectors = None

    def __str__(self) -> str:
        return '<<{} {}>>'.format(self.__class__.__name__, self.name)

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

    def _bare_model_path(self) -> Path:
        return (self.result_dir / self.name).with_suffix('.bare-model')

    def _model_path(self) -> Path:
        return (self.result_dir / self.name).with_suffix('.model')

    def _vectors_path(self) -> Path:
        return (self.result_dir / self.name).with_suffix('.vec')

    def _load_model(self):
        if self._model is not None:
            return self._model
        LOGGER.debug('Loading model for {} from {}'.format(self, self.model_path()))
        self._model = FastText.load(str(self.model_path()))
        self._vectors = self._model.wv.vectors
        return self._model

    def _load_vectors(self):
        if self._vectors is not None:
            return self._vectors
        LOGGER.debug('Loading vectors for {} from {}'.format(self, self.vectors_path()))
        self._vectors = KeyedVectors.load_word2vec_format(str(self.vectors_path()))
        return self._vectors

    def _build_vocab(self) -> FastText:
        try:
            with self._bare_model_path().open('rb') as rf:
                LOGGER.debug('Loading vocab for {} from {}'.format(self, self._bare_model_path()))
                saved_values = pickle.load(rf)
        except IOError:
            bare_model = FastText(**self.fasttext_parameters)
            build_vocab_parameters = {
                **FASTTEXT_PARAMETERS['build_vocab'],
                **{
                    key: value
                    for (key, value) in self.fasttext_parameters.items()
                    if key in FASTTEXT_PARAMETERS['build_vocab'].keys()
                },
            }
            LOGGER.debug('Building vocab for {}'.format(self))
            bare_model.build_vocab(corpus_iterable=self.corpus, **build_vocab_parameters)
            saved_values = {'model_values': {}, 'wv_values': {}}
            for key in FASTTEXT_PARAMETERS['build_vocab_keys']:
                if key in vars(bare_model):
                    saved_values['model_values'][key] = bare_model.__dict__[key]
                elif key in vars(bare_model.wv):
                    saved_values['wv_values'][key] = bare_model.wv.__dict__[key]
                else:
                    message = 'Key {} not found in FastText model or its keyed vectors'.format(key)
                    raise KeyError(message)
            with self._bare_model_path().open('wb') as wf:
                LOGGER.debug('Saving vocab for {} to {}'.format(self, self._bare_model_path()))
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
        pass
