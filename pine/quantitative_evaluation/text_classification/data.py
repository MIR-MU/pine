# -*- coding: utf-8 -*-

from __future__ import annotations

from itertools import chain
from logging import getLogger
from pathlib import Path
from typing import List

import numpy as np
from scipy.io import loadmat

from ...configuration import TEXT_CLASSIFICATION_DATASET_SIZES


LOGGER = getLogger(__name__)


class Document:
    def __init__(self, words: List[str], target: int):
        words = tuple(words)
        for word in words:
            if not isinstance(word, str):
                raise ValueError('Word {} is not a string'.format(word))
        self.words = words
        if not isinstance(target, int):
            raise ValueError('Target {} is not an integer'.format(target))
        self.target = target


class Dataset:
    def __init__(self, name: str, path: Path, split_idx: int):
        if name not in TEXT_CLASSIFICATION_DATASET_SIZES:
            known_datasets = ', '.join(TEXT_CLASSIFICATION_DATASET_SIZES)
            message = 'Unknown dataset {} (known datasets: {})'.format(name, known_datasets)
            raise ValueError(message)
        self.name = name
        self.split_idx = split_idx
        self.path = path
        self.loaded = False

    def __str__(self) -> str:
        return 'dataset {}, split {}'.format(self.name, self.split_idx)

    def __hash__(self) -> int:
        return hash((self.name, self.split_idx))

    def __eq__(self, other: Dataset) -> bool:
        return self.name == other.name and self.split_idx == other.split_idx

    def load(self):
        if self.loaded:
            return
        self.train_documents = []
        self.test_documents = []
        LOGGER.debug('Loading {}'.format(self))
        data = loadmat(str(self.path))
        self._train_docs_per_split(data)
        if len(self.train_documents) != TEXT_CLASSIFICATION_DATASET_SIZES[self.name]:
            message = 'Expected {} train documents but loaded {}'
            message = message.format(self.train_document_sizes[self.name], len(self.train_documents))
            raise ValueError(message)
        self.loaded = True

    def _train_docs_per_split(self, data: np.ndarray):
        def isstr(s):
            return isinstance(s, str)

        def extract_document(args):
            words, frequencies = args
            words = list(filter(isstr, chain(*words[0])))
            frequencies = list(map(int, frequencies[0]))
            if len(words) != len(frequencies):
                message = 'Different number of words ({}) and word frequencies ({})'
                raise ValueError(message.format(len(words), len(frequencies)))
            document = [frequency * [word] for word, frequency in zip(words, frequencies)]
            document = list(chain(*document))
            return document

        is_split_format = (
            'TR' in data and
            'TE' in data and
            ('the_words' in data or 'words' in data) and
            'BOW_X' in data
        )
        is_unsplit_format = (
            'words_tr' in data and
            'words_te' in data and
            'ytr' in data and
            'yte' in data and
            'BOW_xtr' in data and
            'BOW_xte' in data
        )

        if is_split_format:
            train_idxs = set(data['TR'][self.split_idx])
            test_idxs = set(data['TE'][self.split_idx])
            if train_idxs & test_idxs:
                raise ValueError('Train and test splits overlap')

            documents = data['the_words'] if 'the_words' in data else data['words']
            documents = list(filter(len, documents[0]))
            frequencies = data['BOW_X'][0]
            if len(documents) != len(frequencies):
                message = 'Different number of documents ({}) and word frequencies ({})'
                raise ValueError(message.format(len(documents), len(frequencies)))
            documents = zip(documents, frequencies)
            documents = list(map(extract_document, documents))

            targets = data['Y'][0]
            targets = list(map(int, targets))
            if len(documents) != len(targets):
                message = 'Different length of documents ({}) and targets ({})'
                raise ValueError(message.format(len(documents), len(targets)))

            for document_id, (document, target) in enumerate(zip(documents, targets)):
                document_id += 1
                document = Document(document, target)
                if document_id in train_idxs:
                    train_idxs.remove(document_id)
                    self.train_documents.append(document)
                elif document_id in test_idxs:
                    test_idxs.remove(document_id)
                    self.test_documents.append(document)
                else:
                    message = 'Document id {} is not present in either split'
                    raise ValueError(message.format(document_id))
        elif is_unsplit_format:
            train_documents = list(filter(len, data['words_tr'][0]))
            train_frequencies = data['BOW_xtr'][0]
            if len(train_documents) != len(train_frequencies):
                message = 'Different number of train documents ({}) and word frequencies ({})'
                raise ValueError(message.format(len(train_documents), len(train_frequencies)))
            train_documents = zip(train_documents, train_frequencies)
            train_documents = list(map(extract_document, train_documents))

            test_documents = list(filter(len, data['words_te'][0]))
            test_frequencies = data['BOW_xte'][0]
            if len(test_documents) != len(test_frequencies):
                message = 'Different number of test documents ({}) and word frequencies ({})'
                raise ValueError(message.format(len(test_documents), len(test_frequencies)))
            test_documents = zip(test_documents, test_frequencies)
            test_documents = list(map(extract_document, test_documents))

            train_targets = list(map(int, data['ytr'][0]))
            test_targets = list(map(int, data['yte'][0]))
            if len(train_documents) != len(train_targets):
                message = 'Different length of train documents ({}) and targets ({})'
                raise ValueError(message.format(len(train_documents), len(train_targets)))
            if len(test_documents) != len(test_targets):
                message = 'Different length of test documents ({}) and targets ({})'
                raise ValueError(message.format(len(test_documents), len(test_targets)))

            for document, target in zip(train_documents, train_targets):
                document = Document(document, target)
                self.train_documents.append(document)
            for document, target in zip(test_documents, test_targets):
                document = Document(document, target)
                self.test_documents.append(document)
        else:
            raise ValueError('Unrecognized matrix format')


def load_kusner_datasets(path: Path) -> List[Dataset]:
    name = path.name.split('-')[0]
    if '_split' in name:
        datasets = [Dataset(name, path, split_idx) for split_idx in range(5)]
    else:
        datasets = [Dataset(name, path, 0)]
    return datasets
