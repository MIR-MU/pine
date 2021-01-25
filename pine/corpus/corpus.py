# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from tqdm import tqdm

from ..configuration import CORPUS_SIZES


def get_corpus(name: str, result_dir: Path, language: str = 'en') -> Corpus:
    if name not in CORPUS_SIZES:
        known_corpora = ', '.join(CORPUS_SIZES)
        message = 'Unknown corpus {} (known corpora: {})'.format(name, known_corpora)
        raise ValueError(message)
    if language not in CORPUS_SIZES[name]:
        known_languages = ', '.join(CORPUS_SIZES[name])
        message = 'Unknown language {} for corpus {} (known languages: {})'
        message = message.format(language, name, known_languages)
        raise ValueError(message)
    corpus_size = CORPUS_SIZES[name][language]

    if name == 'wikipedia':
        from .wikipedia import get_corpus_path
    elif name == 'common_crawl':
        from .common_crawl import get_corpus_path
    else:
        raise ValueError('Corpus {} not yet implemented'.format(name))

    corpus_path = get_corpus_path(language=language, name=name, result_dir=result_dir)
    corpus = LineSentence(name, corpus_path, corpus_size)
    return corpus


Corpus = Iterable[Iterable[str]]


class LineSentence:
    def __init__(self, name: str, path: Path, size: int):
        self.name = name
        self.path = path
        self.size = size

    def __iter__(self) -> Iterable[List[str]]:
        with self.path.open('rt') as f:
            sentences = tqdm(f, desc='Reading {}'.format(self), total=self.size)
            for sentence in sentences:
                sentence = sentence.split()
                yield sentence

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return '<<{} {}>>'.format(self.__class__.__name__, self.name)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Corpus) -> bool:
        return self.name == other.name
