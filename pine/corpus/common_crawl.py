# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List
from multiprocessing import Pool

from ..util import simple_preprocess, download_to
from ..configuration import CORPORA
from tqdm import tqdm


def get_corpus_path(language: str, name: str, corpus_dir: Path) -> Path:
    if language != 'en':
        raise ValueError('Unsupported Common Crawl language {}'.format(language))
    corpus_dir /= name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = corpus_dir / language
    corpus_path = corpus_path.with_suffix('.txt')
    if corpus_path.exists():
        return corpus_path
    with corpus_path.open('wt') as f:
        sentences = EnglishCommonCrawlSentences('Creating corpus {}'.format(corpus_path), corpus_dir)
        for sentence in sentences:
            sentence = ' '.join(sentence)
            print(sentence, file=f)
    return corpus_path


class EnglishCommonCrawlSentences:
    def __init__(self, desc: str, corpus_dir: Path):
        self.desc = desc
        self.corpus_dir = corpus_dir

    def __iter__(self) -> Iterable[List[str]]:
        shards = CORPORA['common_crawl']['en']
        shard_paths = []
        for shard_number, shard in enumerate(shards):
            shard_path = '{:02d}.txt'.format(shard_number)
            shard_path = self.corpus_dir / shard_path
            download_to(path=shard_path, **shard)
            shard_paths.append(shard_path)

        print([
            {
                'url': shard['url'],
                'size': shard_path.stat().st_size,
            }
            for shard, shard_path
            in zip(shards, shard_paths)
        ])

        with Pool(None) as pool:
            shard_paths = tqdm(shard_paths, desc=self.desc)
            for shard_path in shard_paths:
                with shard_path.open('rt') as f:
                    sentences = pool.imap(simple_preprocess, f)
                    sentences = filter(lambda x: x, sentences)
                    for sentence in sentences:
                        yield sentence
                shard_path.unlink()
