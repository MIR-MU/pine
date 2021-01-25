# -*- coding: utf-8 -*-

from __future__ import annotations

from logging import getLogger
from typing import Dict, Literal, Union, List
from pathlib import Path
import tarfile

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError('Please install PyTorch to evaluate on language modeling')

from ..config import LANGUAGE_MODELING_DATASETS
from ..language_model import LanguageModel
from ..util import download_to


LOGGER = getLogger(__name__)


def get_dataset_paths(language: str, result_dir: Path) -> Dataset:
    readme_path = result_dir / 'icml14-data.readme'
    if not readme_path.exists():
        dataset_tarfile_path = (result_dir / 'icml14-data').with_suffix('.tar.bz2')
        download_to(path=dataset_tarfile_path, **LANGUAGE_MODELING_DATASETS)
        LOGGER.info('Extracting datasets from {} to {}'.format(dataset_tarfile_path, result_dir))
        with tarfile.open(dataset_tarfile_path, 'r:bz2') as tf:
            tf.extractall(result_dir)
        dataset_tarfile_path.unlink()

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


def evaluate(dataset_paths: Dataset, language_model: LanguageModel,
             result_dir: Path, seed: int = 21, device: str = 'cuda') -> LanguageModelingResults:
    pass


Dataset = Dict[Literal['vocab', 'train', 'validation', 'test'], Path]
LanguageModelingResults = Union[
    Dict[Literal['train', 'validation'], List[float]],
    Dict[Literal['test'], float]
]
