# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from gensim.utils import simple_preprocess
import smart_open
from tqdm import tqdm


SHARD_URL = 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.{:02d}.deduped.xz'


def get_corpus_path(language: str, name: str, result_dir: Path) -> Path:
    if language != 'en':
        raise ValueError('Unsupported Common Crawl language {}'.format(language))
    corpus_path = (result_dir / name).with_suffix('.txt')
    if corpus_path.exists():
        return corpus_path
    with corpus_path.open('wt') as f:
        sentences = EnglishCommonCrawlSentences('Creating corpus {}'.format(corpus_path))
        for sentence in sentences:
            sentence = ' '.join(sentence)
            print(sentence, file=f)
    return corpus_path


class EnglishCommonCrawlSentences:
    def __init__(self, desc: str):
        self.desc = desc

    def __iter__(self) -> Iterable[List[str]]:
        shards = [SHARD_URL.format(shard_number) for shard_number in range(100)]
        shards = tqdm(shards, desc=self.desc, position=0)
        for shard in shards:
            with smart_open.open(shard, 'rt') as f:
                sentences = tqdm(f, position=1, leave=False)
                sentences = map(simple_preprocess, f)
                sentences = filter(len, sentences)
                for sentence in sentences:
                    yield sentence