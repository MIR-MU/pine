# -*- coding: utf-8 -*-

from __future__ import annotations

from itertools import chain
from multiprocessing import Pool
from typing import Iterable, Dict, List

from gensim.utils import simple_preprocess
import gensim.downloader
from tqdm import tqdm


def get_corpus_path(language, result_dir):
    if language != 'en':
        raise ValueError('Unsupported wikipedia language {}'.format(language))
    corpus_path = (result_dir / 'wikipedia').with_suffix('.txt')
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
        sentences = (sentence for sentence in sentences if sentence)
        sentences = chain([section_title], sentences)
        sentences = map(simple_preprocess, sentences)
        all_sentences.extend(sentences)
    return all_sentences


class EnglishWikipediaSentences(object):
    def __init__(self, desc: str, percentage: float = 1.0):
        self.desc = desc
        self.percentage = percentage
        self.iterable = None

    def __iter__(self) -> EnglishWikipediaSentences:
        self.__init__(self.desc, self.percentage)
        return self

    def _read_sentences(self, corpus_num_articles: int = 4924894) -> Iterable[str]:
        total = int(corpus_num_articles * self.percentage)
        articles = gensim.downloader.load('wiki-english-20171001')
        articles = (article for article, _ in zip(articles, range(total)))
        articles = tqdm(articles, desc=self.desc, total=total)
        with Pool(None) as pool:
            for sentences in pool.imap_unordered(_read_sentences_helper, articles):
                for sentence in sentences:
                    yield sentence

    def __next__(self) -> str:
        if self.iterable is None:
            self.iterable = self._read_sentences()
        corpus_sentence = next(self.iterable)
        return corpus_sentence
