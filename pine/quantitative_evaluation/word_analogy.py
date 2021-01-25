# -*- coding: utf-8 -*-

import json
from pathlib import Path

from ..config import WORD_ANALOGY_PARAMETERS, WORD_ANALOGY_DATASETS
from ..language_model import LanguageModel
from ..util import download_to


def get_dataset_path(language: str, result_dir: Path) -> Path:
    if language not in WORD_ANALOGY_DATASETS:
        known_languages = ', '.join(WORD_ANALOGY_DATASETS)
        message = 'Unknown language {} for word analogies (known languages: {})'
        message = message.format(known_languages)
        raise ValueError(message)
    url = WORD_ANALOGY_DATASETS[language]['url']
    size = WORD_ANALOGY_DATASETS[language]['size']

    dataset_path = result_dir / 'word-analogy-{}'.format(language)
    dataset_path = dataset_path.with_suffix('.txt')
    if dataset_path.exists():
        return dataset_path

    download_to(url, size, dataset_path)
    return dataset_path


def evaluate(dataset_path: Path, language_model: LanguageModel, result_dir: Path) -> float:
    result_filename = '{}-{}'.format(language_model.name, dataset_path.stem)
    result_path = result_dir / result_filename
    result_path = result_path.with_suffix('.json')
    try:
        with result_path.open('rt') as rf:
            results = json.load(rf)
    except IOError:
        results = language_model.model.wv.evaluate_word_analogies(
            str(dataset_path),
            **WORD_ANALOGY_PARAMETERS,
        )
        with result_path.open('wt') as wf:
            json.dump(results, wf, indent=4, sort_keys=True)
    score, sections = results
    return score