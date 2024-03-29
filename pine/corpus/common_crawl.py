# -*- coding: utf-8 -*-

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Iterable, List
from multiprocessing import Pool, Semaphore

from ..util import download_to, parallel_simple_preprocess
from ..configuration import CORPORA, IO_QUEUE_SIZE
from tqdm import tqdm


def get_corpus_path(language: str, name: str, corpus_dir: Path) -> Path:
    if language not in CORPORA['common_crawl']:
        known_languages = ', '.join(CORPORA['common_crawl'])
        message = 'Unsupported Common Crawl language {} (known languages: {})'
        message = message.format(language, known_languages)
        raise ValueError(message)
    corpus_dir /= name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / language
    corpus_path = corpus_path.with_suffix('.txt.gz')
    if corpus_path.exists():
        return corpus_path
    with gzip.open(corpus_path, 'wt', compresslevel=9) as f:
        desc = 'Creating corpus {}'.format(corpus_path)
        semaphore = Semaphore(IO_QUEUE_SIZE)
        sentences = CommonCrawlSentences(desc, semaphore, corpus_dir, language)
        for sentence in sentences:
            sentence = ' '.join(sentence)
            print(sentence, file=f)
    return corpus_path


class CommonCrawlSentences:
    def __init__(self, desc: str, semaphore, corpus_dir: Path, language: str):
        self.desc = desc
        self.semaphore = semaphore
        self.corpus_dir = corpus_dir
        self.language = language

    def __iter__(self) -> Iterable[List[str]]:
        shards = CORPORA['common_crawl'][self.language]
        shard_paths = []
        for shard_number, shard in enumerate(shards):
            shard_path = '{:02d}.txt'.format(shard_number)
            shard_path = self.corpus_dir / shard_path
            if not shard_path.exists():
                download_to(path=shard_path, **shard)
            shard_paths.append(shard_path)

        with Pool(None) as pool:
            shard_paths = tqdm(shard_paths, desc=self.desc)
            for shard_path in shard_paths:
                sentences = parallel_simple_preprocess(pool, shard_path, self.semaphore)
                for sentence in sentences:
                    if sentence:
                        yield sentence
                    self.semaphore.release()
                shard_path.unlink()
