# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Iterable, Optional
from pathlib import Path

try:
    import torch
except ImportError:
    raise ImportError('For training language models, please install PyTorch')

from gensim.utils import tokenize

from .language_modeling import Dataset


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, dataset: Dataset, dictionary: Optional[Dictionary] = None):
        self.train, self.valid, self.test = None, None, None
        self.dictionary = dictionary
        self.dataset = dataset

    def tokenize_all(self):
        self.train = self.tokenize('train')
        self.valid = self.tokenize('validation')
        self.test = self.tokenize('test')

    def tokenize(self, subset: str) -> torch.Tensor:
        ids = []
        for word in read_text_file(self.dataset[subset]):
            if word in self.dictionary.word2idx:
                ids.append(self.dictionary.word2idx[word])
        ids = torch.tensor(ids).type(torch.int64)

        return ids


def simple_preprocess(text: str) -> List[str]:
    return tokenize(text, lower=True, deacc=False, errors='ignore')


def read_text_file(path: Path) -> Iterable[str]:
    with path.open('rt') as f:
        for line in map(simple_preprocess, f):
            for word in line:
                yield word
            yield '<eos>'
