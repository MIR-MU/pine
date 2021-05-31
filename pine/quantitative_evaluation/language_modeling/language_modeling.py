# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Dict, Tuple, Sequence, Any
from pathlib import Path
from matplotlib.figure import Figure

from ...configuration import LANGUAGE_MODELING_DATASETS, JSON_DUMP_PARAMETERS
from ...language_model import LanguageModel
from ...util import download_to, unzip_to


def get_dataset_paths(language: str, dataset_dir: Path) -> Dataset:
    dataset_path = dataset_dir / 'language_modeling'
    dataset_path.mkdir(parents=True, exist_ok=True)
    dataset_readme_path = dataset_path / 'icml14-data.readme'
    if not dataset_readme_path.exists():
        dataset_tarfile_path = (dataset_path / 'icml14-data').with_suffix('.tar')
        download_to(path=dataset_tarfile_path, **LANGUAGE_MODELING_DATASETS)
        unzip_to(dataset_tarfile_path, dataset_path, unlink_after=True)

    dataset_paths = {
        dirname.name.split('-')[-1]: dirname
        for dirname
        in dataset_path.glob('*-{}/'.format(language))
    }

    if language not in dataset_paths:
        known_languages = ', '.join(dataset_paths)
        message = 'Unknown language {} for language modeling (known languages: {})'
        message = message.format(known_languages)
        raise ValueError(message)
    dataset_path = dataset_paths[language]
    dataset_paths = {
        'vocab': dataset_path / '1m-mono' / 'vocab',
        'train': (dataset_path / '1m-mono' / 'train').with_suffix('.in'),
        'validation': (dataset_path / '1m-mono' / 'test').with_suffix('.in'),
        'test': (dataset_path / '1m-mono' / 'finaltest').with_suffix('.in'),
    }
    return dataset_paths


def evaluate(dataset_paths: Dataset, language_model: LanguageModel) -> Result:
    result_path = language_model.model_dir / 'language_modeling.json'
    try:
        with result_path.open('rt') as rf:
            result = json.load(rf)
    except IOError:
        from .training import train_and_evaluate
        result = train_and_evaluate(dataset_paths, language_model)
        with result_path.open('wt') as wf:
            json.dump(result, wf, **JSON_DUMP_PARAMETERS)
    return Result(language_model, result)


Dataset = Dict[str, Path]
Perplexity = float
Loss = float
LearningRate = float
EvaluationResult = Tuple[Perplexity, Loss]
TrainingResult = Sequence[EvaluationResult]
ValidationResult = EvaluationResult
TestResult = EvaluationResult
EpochResult = Tuple[TrainingResult, ValidationResult, LearningRate]
RawResult = Tuple[TestResult, Sequence[EpochResult]]


class Result:
    """The results of a language modeling task for a log-bilinear language model.

    Parameters
    ----------
    language_model : :class:`~pine.language_model.LanguageModel`
        A log-bilinear language model.
    result : RawResult
        Results of a language modeling task.

    Attributes
    ----------
    language_model : :class:`~pine.language_model.LanguageModel`
        A log-bilinear language model.
    result : RawResult
        Results of a language modeling task.

    """
    def __init__(self, language_model: LanguageModel, result: RawResult):
        self.language_model = language_model
        self.result = result

    def plot(self, *args: Any, **kwargs: Any) -> 'Figure':
        from .view import plot_language_modeling_results
        return plot_language_modeling_results(self.language_model, *args, **kwargs)

    def __repr__(self) -> str:
        test_result, _ = self.result
        test_perplexity, _ = test_result
        return 'Test perplexity of {}: {:.2f}'.format(self.language_model, test_perplexity)

    def _repr_html_(self) -> str:
        figure = self.plot()
        return figure._repr_html_()
