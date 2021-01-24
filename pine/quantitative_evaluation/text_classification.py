# -*- coding: utf-8 -*-

from __future__ import annotations

from itertools import chain
from functools import lru_cache, partial
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
from zipfile import ZipFile

from gensim.corpora import Dictionary
from gensim.interfaces import SimilarityABC
from gensim.models import TfidfModel, KeyedVectors
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import numpy as np
from scipy.io import loadmat
from scipy.stats import mode
import smart_open
from tqdm import tqdm

from .. import LanguageModel
from ..config import TEXT_CLASSIFICATION_DATASET_SIZES, TEXT_CLASSIFICATION_DATASETS, TEXT_CLASSIFICATION_METHODS


COMMON_VECTORS = None
EXECUTOR = ProcessPoolExecutor(None)
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


def __wmdistance(query: List[str], document: List[str]) -> float:
    return COMMON_VECTORS.wmdistance(query, document)


@lru_cache(maxsize=None)
def _wmdistance(query: List[str], document: List[str]) -> float:
    return EXECUTOR.submit(__wmdistance, query, document)


def wmdistance(query: List[str], document: List[str]) -> float:
    if query < document:  # Enforce symmetric caching
        distance = _wmdistance(query, document)
    else:
        distance = _wmdistance(document, query)
    return distance


class WmdSimilarity(SimilarityABC):
    def __init__(self, corpus: List[List[str]], vectors: KeyedVectors, num_best: Optional[int] = None,
                 chunksize: int = 256):
        self.corpus = corpus
        self.vectors = vectors
        self.num_best = num_best
        self.chunksize = chunksize
        self.normalize = False
        self.index = np.arange(len(corpus))

    def __len__(self) -> int:
        return len(self.corpus)

    def get_similarities(self, queries: List[List[str]]) -> np.ndarray:
        global COMMON_VECTORS

        COMMON_VECTORS = self.vectors

        result = []
        for query in tqdm(queries):
            qresult = [wmdistance(query, document) for document in self.corpus]
            qresult = 1. / (1. + np.array([future.result() for future in qresult]))
            result.append(qresult)
        result = np.array(result)

        COMMON_VECTORS = None

        return result


class Evaluator:
    def __init__(self, dataset: Dataset, model: LanguageModel, method: str):
        self.dataset = dataset
        if method not in TEXT_CLASSIFICATION_METHODS:
            known_methods = ', '.join(TEXT_CLASSIFICATION_METHODS)
            message = 'Unknown method {} (known methods: {})'.format(method, known_methods)
            raise ValueError(message)
        self.method = method
        self.vectors = model.vectors

    def __hash__(self) -> int:
        return hash((self.dataset, self.method))

    def __eq__(self, other: Evaluator) -> bool:
        return self.dataset == other.dataset and self.method == other.method

    @lru_cache(maxsize=None)
    def _preprocess_dataset(self, level: str) -> Tuple[
                List[Document], List[Document], SimilarityABC, List[Tuple[int, float]]
            ]:
        LOGGER.debug('Preprocessing {} ({})'.format(self.dataset, level))

        if level == 'validation':
            pivot = int(round(len(self.dataset.train_documents) * 0.8))
            train_documents = self.dataset.train_documents[:pivot]
            test_documents = self.dataset.train_documents[pivot:]
        elif level == 'test':
            train_documents = self.dataset.train_documents
            test_documents = self.dataset.test_documents
        else:
            message = 'Expected validation or test level, but got {}'
            raise ValueError(message.format(level))

        if self.method == 'scm':
            train_corpus = [document.words for document in train_documents]
            dictionary = Dictionary(train_corpus, prune_at=None)
            tfidf = TfidfModel(dictionary=dictionary, smartirs='nfn')
            termsim_index = WordEmbeddingSimilarityIndex(self.vectors)
            similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)
            train_corpus = [dictionary.doc2bow(document) for document in train_corpus]
            train_corpus = tfidf[train_corpus]
            similarity_model = SoftCosineSimilarity(train_corpus, similarity_matrix)
            test_corpus = (document.words for document in test_documents)
            test_corpus = [dictionary.doc2bow(document) for document in test_corpus]
            test_corpus = tfidf[test_corpus]
        elif self.method == 'wmd':
            train_corpus = [document.words for document in train_documents]
            similarity_model = WmdSimilarity(train_corpus, self.vectors)
            test_corpus = [document.words for document in test_documents]
        else:
            message = 'Preprocessing for method {} not yet implemented'.format(self.method)
            raise ValueError(message)

        return (train_documents, test_documents, similarity_model, test_corpus)

    def _collect_preds_sequential(self, level: str, knn: int) -> Tuple[List[int], List[int]]:
        train_documents, test_documents, similarity_model, test_corpus = self._preprocess_dataset(level)

        LOGGER.debug('Evaluating {} ({}, k={})'.format(self.dataset, level, knn))

        y_preds = []
        y_trues = []
        with np.errstate(all='ignore'):
            similarities = similarity_model[test_corpus]
        expected_shape = (len(test_documents), len(train_documents))
        if similarities.shape != expected_shape:
            message = 'Expected similarities with shape {}, but received shape {}'
            raise ValueError(message.format(expected_shape, similarities.shape))
        similarities = enumerate(zip(test_documents, similarities))
        for test_doc_id, (test_doc, train_doc_similarities) in similarities:
            most_similar_idxs = train_doc_similarities.argsort()[::-1][:knn]
            voted_targets = [train_documents[idx].target for idx in most_similar_idxs]
            if len(voted_targets) != knn:
                message = 'Expected {} nearest neighbors, but got {}'
                raise ValueError(message.format(knn, len(voted_targets)))
            y_pred = mode(voted_targets).mode[0]
            y_true = test_doc.target

            y_preds.append(y_pred)
            y_trues.append(y_true)
        return y_preds, y_trues

    def _evaluate(self, knn: int, level: str = 'validation') -> float:
        y_preds, y_trues = self._collect_preds_sequential(level, knn)
        test_errors = sum(y_pred != y_true for y_pred, y_true in zip(y_preds, y_trues))
        error_rate = test_errors / len(y_preds)
        return error_rate

    def _get_best_knn(self, knns: Tuple[int] = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)) -> int:
        self._preprocess_dataset('validation')
        best_knn, best_error_rate = None, float('inf')
        worst_knn, worst_error_rate = None, float('-inf')
        for knn, error_rate in zip(knns, map(self._evaluate, knns)):
            if error_rate < best_error_rate:
                best_knn, best_error_rate = knn, error_rate
            if error_rate > worst_error_rate:
                worst_knn, worst_error_rate = knn, error_rate
        message = 'Best/worst value of k for {}: {}/{} (validation error rate: {:.2f}/{:.2f}%)'
        message = message.format(self.dataset, best_knn, worst_knn,
                                 best_error_rate * 100.0, worst_error_rate * 100.0)
        LOGGER.debug(message)
        return best_knn

    def evaluate(self) -> float:
        self.dataset.load()
        knn = self._get_best_knn()
        error_rate = self._evaluate(knn, level='test')
        message = 'Test error rate for {}: {:.2f}%'.format(self.dataset, error_rate * 100.0)
        LOGGER.debug(message)
        return error_rate


def load_kusner_datasets(path: Path) -> List[Dataset]:
    name = path.name.split('-')[0]
    if '_split' in name:
        datasets = [Dataset(name, path, split_idx) for split_idx in range(5)]
    else:
        datasets = [Dataset(name, path, 0)]
    return datasets


def print_error_rate_analysis(dataset: Dataset, error_rates: List[float]):
    error_rate = np.mean(error_rates) * 100.0
    if len(error_rates) > 1:
        sem = np.std(error_rates, ddof=1) * 100.0 / np.sqrt(len(error_rates))
        ci = 1.96 * sem
        message = 'Test error rate for dataset {}: {:.2f}% (SEM: {:g}%, 95% CI: ±{:g}%)'
        LOGGER.info(message.format(dataset.name, error_rate, sem, ci))
    else:
        message = 'Test error rate for dataset {}: {:.2f}%'
        LOGGER.info(message.format(dataset.name, error_rate))


def evaluate(dataset_path: Path, language_model: LanguageModel,
             method: str, result_dir: Path) -> List[float]:
    datasets = load_kusner_datasets(dataset_path)
    dataset, *_ = datasets
    result_filename = '{}-{}-{}.txt'.format(method, language_model.name, dataset.name)
    result = result_dir / Path(result_filename)
    try:
        with result.open('rt') as f:
            error_rates = [float(line) for line in f]
    except IOError:
        error_rates = [Evaluator(dataset, language_model, method).evaluate() for dataset in datasets]
        with result.open('wt') as f:
            for line in error_rates:
                print(line, file=f)
    print_error_rate_analysis(dataset, error_rates)
    _wmdistance.cache_clear()  # Clear the WMD cache, so that the datasets don't take up RAM

    return error_rates


def get_dataset_paths(result_dir: Path, buffer_size: int = 2**20) -> List[Path]:
    dataset_paths = result_dir.glob('*.mat')
    dataset_paths = sorted(dataset_paths)
    if dataset_paths:
        return dataset_paths

    dataset_zipfile_path = (result_dir / 'WMD_datasets').with_suffix('.zip')
    desc = 'Downloading datasets {}'.format(dataset_zipfile_path)
    with tqdm(total=TEXT_CLASSIFICATION_DATASETS['size'], unit='B', desc=desc) as pbar:
        with dataset_zipfile_path.open('wb') as wf:
            with smart_open.open(TEXT_CLASSIFICATION_DATASETS['url'], 'rb') as rf:
                for data in iter(partial(rf.read, buffer_size), b''):
                    wf.write(data)
                    pbar.update(len(data))
    LOGGER.info('Extracting datasets from {} to {}'.format(dataset_zipfile_path, result_dir))
    with ZipFile(dataset_zipfile_path, 'r') as zf:
        zf.extractall(result_dir)
    dataset_zipfile_path.unlink()
    (result_dir / '20ng2_500-emd_tr_te.mat').unlink()

    dataset_paths = result_dir.glob('*.mat')
    dataset_paths = sorted(dataset_paths)
    return dataset_paths
