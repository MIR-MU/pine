# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import lru_cache
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, Future
import shelve

from gensim.corpora import Dictionary
from gensim.interfaces import SimilarityABC
from gensim.models import TfidfModel, KeyedVectors
from gensim.similarities import SoftCosineSimilarity, SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex
import numpy as np
from scipy.stats import mode
from tqdm import tqdm

from ...language_model import LanguageModel
from ...configuration import TEXT_CLASSIFICATION_METHOD_PARAMETERS
from .data import Dataset, Document


COMMON_VECTORS = None
EXECUTOR = ProcessPoolExecutor(None)
LOGGER = getLogger(__name__)


class Evaluator:
    def __init__(self, dataset: Dataset, model: LanguageModel, method: str):
        self.dataset = dataset
        if method not in TEXT_CLASSIFICATION_METHOD_PARAMETERS:
            known_methods = ', '.join(TEXT_CLASSIFICATION_METHOD_PARAMETERS)
            message = 'Unknown method {} (known methods: {})'.format(method, known_methods)
            raise ValueError(message)
        self.model = model
        self.method = method

    def __hash__(self) -> int:
        return hash((self.dataset, self.method))

    def __eq__(self, other: Evaluator) -> bool:
        return self.dataset == other.dataset and self.method == other.method

    @lru_cache(maxsize=1)
    def _preprocess_dataset(self, level: str) -> Tuple[
                List[Document], List[Document], np.ndarray, List[Tuple[int, float]]
            ]:
        LOGGER.info('Preprocessing {} ({})'.format(self.dataset, level))

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

        cache_path = self.model.cache_dir / 'text_classification'
        cache_path.mkdir(exist_ok=True)
        method_parameters = TEXT_CLASSIFICATION_METHOD_PARAMETERS[self.method]
        if self.method == 'scm':
            train_corpus = [document.words for document in train_documents]
            dictionary = Dictionary(train_corpus, prune_at=None)
            tfidf = TfidfModel(dictionary=dictionary, smartirs='nfn')
            termsim_index = WordEmbeddingSimilarityIndex(self.model.vectors,
                                                         **method_parameters['similarity_index'])
            cache_path = cache_path / '{}-{}-{}'.format(self.dataset.name, self.method, level)
            try:
                SparseTermSimilarityMatrix.load(str(cache_path))
            except IOError:
                similarity_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf,
                                                               **method_parameters['similarity_matrix'])
                similarity_matrix.matrix.eliminate_zeros()  # Apply fix from Gensim issue #2783
                similarity_matrix.save(str(cache_path))
            train_corpus = [dictionary.doc2bow(document) for document in train_corpus]
            train_corpus = tfidf[train_corpus]
            similarity_model = SoftCosineSimilarity(train_corpus, similarity_matrix)
            test_corpus = (document.words for document in test_documents)
            test_corpus = [dictionary.doc2bow(document) for document in test_corpus]
            test_corpus = tfidf[test_corpus]
        elif self.method == 'wmd':
            train_corpus = [document.words for document in train_documents]
            cache_path = cache_path / '{}-{}'.format(self.dataset.name, self.method)
            cache_path = cache_path.with_suffix('.shelf')
            similarity_model = ParallelCachingWmdSimilarity(train_corpus, self.model.vectors, cache_path)
            test_corpus = [document.words for document in test_documents]
        else:
            message = 'Preprocessing for method {} not yet implemented'.format(self.method)
            raise ValueError(message)

        with np.errstate(all='ignore'):
            similarities = similarity_model[test_corpus]
        expected_shape = (len(test_documents), len(train_documents))
        if similarities.shape != expected_shape:
            message = 'Expected similarities with shape {}, but received shape {}'
            raise ValueError(message.format(expected_shape, similarities.shape))

        return (train_documents, test_documents, similarities, test_corpus)

    def _collect_preds_sequential(self, level: str, knn: int) -> Tuple[List[int], List[int]]:
        train_documents, test_documents, similarities, test_corpus = self._preprocess_dataset(level)

        y_preds, y_trues = [], []
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
        error_rates = zip(knns, map(self._evaluate, knns))
        error_rates = tqdm(error_rates, desc='Optimizing k for kNN', total=len(knns))
        for knn, error_rate in error_rates:
            if error_rate < best_error_rate:
                best_knn, best_error_rate = knn, error_rate
            if error_rate > worst_error_rate:
                worst_knn, worst_error_rate = knn, error_rate
        message = 'Best/worst value of k for {}: {}/{} (validation error rate: {:.2f}/{:.2f}%)'
        message = message.format(self.dataset, best_knn, worst_knn,
                                 best_error_rate * 100.0, worst_error_rate * 100.0)
        LOGGER.info(message)
        return best_knn

    def evaluate(self) -> float:
        self.dataset.load()
        knn = self._get_best_knn()
        error_rate = self._evaluate(knn, level='test')
        self._preprocess_dataset.cache_clear()
        message = 'Test error rate for {}: {:.2f}%'.format(self.dataset, error_rate * 100.0)
        LOGGER.info(message)
        return error_rate


def wmdistance(query: List[str], document: List[str]) -> float:
    from logging import WARNING
    logger = getLogger('gensim.corpora.dictionary')
    logger.setLevel(WARNING)
    return COMMON_VECTORS.wmdistance(query, document)


class ParallelCachingWmdSimilarity(SimilarityABC):
    def __init__(self, corpus: List[List[str]], vectors: KeyedVectors, cache_path: Path,
                 num_best: Optional[int] = None, chunksize: int = 256):
        self.corpus = corpus
        self.vectors = vectors
        self.cache_path = cache_path
        self.num_best = num_best
        self.chunksize = chunksize
        self.normalize = False
        self.index = np.arange(len(corpus))

    def __len__(self) -> int:
        return len(self.corpus)

    def get_similarities(self, queries: List[List[str]]) -> np.ndarray:
        global COMMON_VECTORS

        COMMON_VECTORS = self.vectors

        with shelve.open(str(self.cache_path), 'c') as shelf:

            def make_symmetric(query: List[str], document: List[str]) -> Tuple[List[str]]:
                if query < document:  # Enforce symmetric caching
                    query, document = document, query
                return (query, document)

            def make_key(query: List[str], document: List[str]) -> str:
                return repr((query, document))

            @lru_cache(maxsize=None)
            def _load_from_shelf(query: List[str], document: List[str]) -> float:
                key = make_key(query, document)
                if key in shelf:
                    return shelf[key]
                return EXECUTOR.submit(wmdistance, query, document)

            def load_from_shelf(query: List[str], document: List[str]) -> float:
                query, document = make_symmetric(query, document)
                return _load_from_shelf(query, document)

            def store_to_shelf(query: List[str], document: List[str], value: float):
                key = make_key(*make_symmetric(query, document))
                if key not in shelf:
                    shelf[key] = value

            result = []
            num_hits, num_misses = 0, 0
            for query in tqdm(queries, desc='Query', position=0):
                futures = [load_from_shelf(query, document) for document in self.corpus]
                distances = []
                documents = tqdm(self.corpus, desc='Document', position=1, leave=False)
                for document, future in zip(documents, futures):
                    if isinstance(future, Future):
                        num_misses += 1
                        distance = future.result()
                        store_to_shelf(query, document, distance)
                    else:
                        num_hits += 1
                        distance = future
                    distances.append(distance)
                similarities = 1. / (1. + np.array(distances))
                result.append(similarities)
                _load_from_shelf.cache_clear()
            result = np.array(result)

            hit_ratio = num_hits * 100.0 / num_misses if num_misses else 0.0
            LOGGER.info('WMD cache hit ratio: {:.2f}%'.format(hit_ratio))

        COMMON_VECTORS = None

        return result
