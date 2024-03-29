# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Tuple, List, Dict
from pathlib import Path

from ..configuration import WORD_ANALOGY_PARAMETERS, WORD_ANALOGY_DATASETS, JSON_DUMP_PARAMETERS
from ..language_model import LanguageModel
from ..util import download_to


def get_dataset_path(language: str, dataset_dir: Path) -> Path:
    if language not in WORD_ANALOGY_DATASETS:
        known_languages = ', '.join(WORD_ANALOGY_DATASETS)
        message = 'Unknown language {} for word analogies (known languages: {})'
        message = message.format(known_languages)
        raise ValueError(message)

    dataset_path = dataset_dir / 'word_analogy'
    dataset_path.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_path / language
    dataset_path = dataset_path.with_suffix('.txt')
    if dataset_path.exists():
        return dataset_path

    download_to(path=dataset_path, **WORD_ANALOGY_DATASETS[language])
    return dataset_path


def evaluate(dataset_path: Path, language_model: LanguageModel) -> Result:
    dataset_path = str(dataset_path)
    result_path = language_model.model_dir / 'word_analogy'
    result_path = result_path.with_suffix('.json')
    try:
        with result_path.open('rt') as rf:
            result = json.load(rf)
    except IOError:
        vectors = language_model.vectors
        result = vectors.evaluate_word_analogies(dataset_path, **WORD_ANALOGY_PARAMETERS)
        with result_path.open('wt') as wf:
            json.dump(result, wf, **JSON_DUMP_PARAMETERS)
    return Result(result, language_model)


TotalAccuracy = float
Category = Dict[str, List[Tuple[str, str, str, str]]]
RawResult = Tuple[TotalAccuracy, List[Category]]


class Result:
    """The results of a word analogy task for a log-bilinear language model.

    Parameters
    ----------
    result : RawResult
        Results of a word analogy task.
    language_model : :class:`~pine.language_model.LanguageModel`
        A log-bilinear language model.

    Attributes
    ----------
    result : RawResult
        Results of a word analogy task.
    language_model : :class:`~pine.language_model.LanguageModel`
        A log-bilinear language model.

    """
    def __init__(self, result: RawResult, language_model: LanguageModel):
        self.result = result
        self.language = language_model.language

    def __repr__(self) -> str:
        total_accuracy, _ = self.result
        return '{:.02f}% ({})'.format(total_accuracy * 100.0, self.language)
