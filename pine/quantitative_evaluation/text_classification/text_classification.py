# -*- coding: utf-8 -*-

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import List

import numpy as np

from ...language_model import LanguageModel
from ...configuration import TEXT_CLASSIFICATION_DATASETS
from ...util import download_to, unzip_to
from .data import load_kusner_datasets


LOGGER = getLogger(__name__)


def get_dataset_paths(language: str, dataset_dir: Path) -> List[Path]:
    if language not in TEXT_CLASSIFICATION_DATASETS:
        known_languages = ', '.join(TEXT_CLASSIFICATION_DATASETS)
        message = 'Unknown language {} for text classification (known languages: {})'
        message = message.format(known_languages)
        raise ValueError(message)

    dataset_path = dataset_dir / 'text_classification'
    dataset_path.mkdir(parents=True, exist_ok=True)
    dataset_paths = dataset_path.glob('*.mat')
    dataset_paths = sorted(dataset_paths)
    if dataset_paths:
        return dataset_paths

    dataset_zipfile_path = (dataset_path / 'WMD_datasets').with_suffix('.zip')
    download_to(path=dataset_zipfile_path, **TEXT_CLASSIFICATION_DATASETS)
    unzip_to(dataset_zipfile_path, dataset_path, unlink_after=True)
    (dataset_path / '20ng2_500-emd_tr_te.mat').unlink()

    dataset_paths = dataset_path.glob('*.mat')
    dataset_paths = sorted(dataset_paths)
    return dataset_paths


def evaluate(dataset_path: Path, language_model: LanguageModel, method: str) -> Result:
    datasets = load_kusner_datasets(dataset_path)
    dataset, *_ = datasets
    result_path = language_model.model_dir / 'text_classification'
    result_path.mkdir(exist_ok=True)
    result_path = result_path / '{}-{}'.format(dataset.name, method)
    result_path = result_path.with_suffix('.txt')
    try:
        with result_path.open('rt') as rf:
            error_rates = [float(line) for line in rf]
    except IOError:
        error_rates = []
    if len(error_rates) < len(datasets):  # Support partially serialized results
        with result_path.open('at') as wf:
            for dataset in datasets[len(error_rates):]:
                from .evaluation import Evaluator
                error_rate = Evaluator(dataset, language_model, method).evaluate()
                error_rates.append(error_rate)
                print(error_rate, file=wf, flush=True)
    return Result(error_rates)


RawResult = List[float]


class Result:
    def __init__(self, result: RawResult):
        self.result = result

    def __repr__(self) -> str:
        error_rate = np.mean(self.result) * 100.0
        if len(self.result) > 1:
            sem = np.std(self.result, ddof=1) * 100.0 / np.sqrt(len(self.result))
            ci = 1.96 * sem
            return '{:.2f}% (SEM: {:.2f}%, 95% CI: ±{:.2f}%)'.format(error_rate, sem, ci)
        else:
            return '{:.2f}%'.format(error_rate)
