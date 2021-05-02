# -*- coding: utf-8 -*-

from __future__ import annotations

import gzip
from itertools import chain
from multiprocessing import Pool, Semaphore
from pathlib import Path
from typing import Iterable, Dict, List

from ..util import produce, simple_preprocess
from ..configuration import CORPORA, IO_QUEUE_SIZE
import gensim.downloader
from tqdm import tqdm


def get_corpus_path(language: str, name: str, corpus_dir: Path) -> Path:
    if language != 'en':
        raise ValueError('Unsupported Wikipedia language {}'.format(language))
    corpus_path = corpus_dir / name
    corpus_path.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_path / language
    corpus_path = corpus_path.with_suffix('.txt.gz')
    if corpus_path.exists():
        return corpus_path
    with gzip.open(corpus_path, 'wt', compresslevel=9) as f:
        desc = 'Creating corpus {}'.format(corpus_path)
        semaphore = Semaphore(IO_QUEUE_SIZE)
        sentences = EnglishWikipediaSentences(desc, semaphore)
        for sentence in sentences:
            sentence = ' '.join(sentence)
            print(sentence, file=f)
    return corpus_path


def _read_sentences_helper(article: Dict[str, str]) -> List[List[str]]:
    all_sentences = []
    for section_title, section_text in zip(article['section_titles'], article['section_texts']):
        sentences = section_text.splitlines()
        sentences = map(str.strip, sentences)
        sentences = filter(len, sentences)
        sentences = chain([section_title], sentences)
        sentences = map(simple_preprocess, sentences)
        all_sentences.extend(sentences)
    return all_sentences


class EnglishWikipediaSentences:
    def __init__(self, desc: str, semaphore, percentage: float = 1.0):
        self.desc = desc
        self.semaphore = semaphore
        self.percentage = percentage

    def __iter__(self) -> Iterable[List[str]]:
        total = int(float(CORPORA['wikipedia']) * self.percentage)
        articles = gensim.downloader.load('wiki-english-20171001')
        articles = (article for article, _ in zip(articles, range(total)))
        articles = tqdm(articles, desc=self.desc, total=total)
        articles = produce(articles, self.semaphore)
        with Pool(None) as pool:
            for sentences in pool.imap_unordered(_read_sentences_helper, articles):
                for sentence in sentences:
                    yield sentence
