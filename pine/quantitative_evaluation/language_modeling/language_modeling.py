# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Dict, Literal, Tuple, Iterable
from pathlib import Path

from ...configuration import LANGUAGE_MODELING_DATASETS, JSON_DUMP_PARAMETERS
from ...language_model import LanguageModel
from ...util import download_to, unzip_to


def get_dataset_paths(language: str, result_dir: Path) -> Dataset:
    readme_path = result_dir / 'icml14-data.readme'
    if not readme_path.exists():
        dataset_tarfile_path = (result_dir / 'icml14-data').with_suffix('.tar.bz2')
        download_to(path=dataset_tarfile_path, **LANGUAGE_MODELING_DATASETS)
        unzip_to(dataset_tarfile_path, result_dir, unlink_after=True)

    dataset_dirnames = {
        dirname.name.split('-')[-1]: dirname
        for dirname
        in result_dir.glob('*-{}/'.format(language))
    }

    if language not in dataset_dirnames:
        known_languages = ', '.join(dataset_dirnames)
        message = 'Unknown language {} for language modeling (known languages: {})'
        message = message.format(known_languages)
        raise ValueError(message)
    dataset_dirname = dataset_dirnames[language]
    dataset_paths = {
        'vocab': dataset_dirname / '1m-mono' / 'vocab',
        'train': (dataset_dirname / '1m-mono' / 'train').with_suffix('.in'),
        'validation': (dataset_dirname / '1m-mono' / 'test').with_suffix('.in'),
        'test': (dataset_dirname / '1m-mono' / 'finaltest').with_suffix('.in'),
    }
    return dataset_paths


def evaluate(dataset_paths: Dataset, language_model: LanguageModel) -> Result:
    result_path = language_model.basename.with_suffix('.language_modeling.json')
    try:
        with result_path.open('rt') as rf:
            result = json.load(rf)
    except IOError:
        from .training import train_and_evaluate
        result = train_and_evaluate(dataset_paths, language_model)
        with result_path.open('wt') as wf:
            json.dump(result, wf, **JSON_DUMP_PARAMETERS)
    return result


Dataset = Dict[Literal['vocab', 'train', 'validation', 'test'], Path]
Perplexity = float
Loss = float
LearningRate = float
EvaluationResult = Tuple[Perplexity, Loss]
TrainingResult = Iterable[EvaluationResult]
ValidationResult = EvaluationResult
TestResult = EvaluationResult
EpochResult = Tuple[TrainingResult, ValidationResult, LearningRate]
Result = Tuple[TestResult, Iterable[EpochResult]]
