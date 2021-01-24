# -*- coding: utf-8 -*-

from __future__ import annotations

from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Dict, List

from gensim.utils import simple_preprocess
import gensim.downloader
from tqdm import tqdm


CORPUS_NUM_ARTICLES = 4924894


def get_corpus_path(language: str, name: str, result_dir: Path) -> Path:
    if language != 'en':
        raise ValueError('Unsupported Wikipedia language {}'.format(language))
    corpus_path = (result_dir / name).with_suffix('.txt')
    if corpus_path.exists():
        return corpus_path
    with corpus_path.open('wt') as f:
        sentences = EnglishWikipediaSentences('Creating corpus {}'.format(corpus_path))
        for sentence in sentences:
            sentence = ' '.join(sentence)
            print(sentence, file=f)
    return corpus_path


def _read_sentences_helper(article: Dict[str, str]) -> List[str]:
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
    def __init__(self, desc: str, percentage: float = 1.0):
        self.desc = desc
        self.percentage = percentage

    def __iter__(self) -> Iterable[List[str]]:
        total = int(CORPUS_NUM_ARTICLES * self.percentage)
        articles = gensim.downloader.load('wiki-english-20171001')
        articles = (article for article, _ in zip(articles, range(total)))
        articles = tqdm(articles, desc=self.desc, total=total)
        with Pool(None) as pool:
            for sentences in pool.imap_unordered(_read_sentences_helper, articles):
                for sentence in sentences:
                    yield sentence
