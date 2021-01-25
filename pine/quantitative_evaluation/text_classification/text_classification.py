# -*- coding: utf-8 -*-

from __future__ import annotations

from logging import getLogger
from pathlib import Path
from typing import List

import numpy as np

from ..language_model import LanguageModel
from ..configuration import TEXT_CLASSIFICATION_DATASETS
from ..util import download_to, unzip_to
from .data import load_kusner_datasets, Dataset


LOGGER = getLogger(__name__)


def get_dataset_paths(dataset_dir: Path) -> List[Path]:
    dataset_path = dataset_dir / 'text_classification'
    dataset_path.mkdir(parents=True, exist_ok=True)
    dataset_paths = dataset_path.glob('*.mat')
    dataset_paths = sorted(dataset_paths)
    if dataset_paths:
        return dataset_paths

    dataset_zipfile_path = (dataset_dir / 'WMD_datasets').with_suffix('.zip')
    download_to(path=dataset_zipfile_path, **TEXT_CLASSIFICATION_DATASETS)
    unzip_to(dataset_zipfile_path, dataset_path, unlink_after=True)
    (dataset_path / '20ng2_500-emd_tr_te.mat').unlink()

    dataset_paths = dataset_dir.glob('*.mat')
    dataset_paths = sorted(dataset_paths)
    return dataset_paths


def evaluate(dataset_path: Path, language_model: LanguageModel, method: str) -> Result:
    datasets = load_kusner_datasets(dataset_path)
    dataset, *_ = datasets
    result_path = language_model.model_dir / 'text_classification'
    dataset_path.mkdir(exist_ok=True)
    result_path = result_path / '{}-{}'.format(dataset.name, method)
    result_path = result_path.with_suffix('.txt')
    try:
        with result_path.open('rt') as rf:
            error_rates = [float(line) for line in rf]
    except IOError:
        error_rates = []
    if len(error_rates) < len(datasets):  # Support partially serialized results
        with result_path.open('wt') as wf:
            for dataset in datasets[len(error_rates):]:
                from .evaluation import Evaluator
                error_rate = Evaluator(dataset, language_model, method).evaluate()
                error_rates.append(error_rate)
                print(error_rate, file=wf, flush=True)
    print_error_rate_analysis(dataset, error_rates)

    return error_rates


Result = List[float]


def print_error_rate_analysis(dataset: Dataset, error_rates: List[float]):
    error_rate = np.mean(error_rates) * 100.0
    if len(error_rates) > 1:
        sem = np.std(error_rates, ddof=1) * 100.0 / np.sqrt(len(error_rates))
        ci = 1.96 * sem
        message = 'Test error rate for dataset {}: {:.2f}% (SEM: {:g}%, 95% CI: ±{:g}%)'
        LOGGER.info(message.format(dataset.name, error_rate, sem, ci))
    else:
        message = 'Test error rate for dataset {}: {:.2f}%'
        LOGGER.info(message.format(dataset.name, error_rate))
