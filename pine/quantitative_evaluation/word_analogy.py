# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Tuple, List, Dict, Literal
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
    result_path = language_model.model_dir / 'word_analogy.json'
    try:
        with result_path.open('rt') as rf:
            result = json.load(rf)
    except IOError:
        vectors = language_model.vectors
        result = vectors.evaluate_word_analogies(dataset_path, **WORD_ANALOGY_PARAMETERS)
        with result_path.open('wt') as wf:
            json.dump(result, wf, **JSON_DUMP_PARAMETERS)
    return result


Result = Tuple[
    float,
    List[
        Dict[
            Literal['correct', 'incorrect'],
            List[Tuple[str, str, str, str]],
        ],
    ],
]
