# -*- coding: utf-8 -*-

from collections import defaultdict

import nltk
import nltk.corpus


nltk.download('brown', quiet=True)


FASTTEXT_PARAMETERS = {
    'baseline': {  # Parameters from paper <https://arxiv.org/abs/1712.09405v1>
        'sg': 0,
        'bucket': 2 * 10**6,
        'negative': 10,
        'alpha': 0.05,
        'min_alpha': 0,
        'sample': 10**-5,
        'min_count': 5,
        'workers': 4,
        'epochs': 1,
        'vector_size': 300,
    },
    'subwords': {
        False: {
            'min_n': 1,
            'max_n': 0,
        },
        True: {
            'min_n': 3,
            'max_n': 6,
        },
    },
    'positions': {
        False: {
            'window': 5,
        },
        'full': {
            'position_dependent_weights': 1,
            'window': 15,
        },
        'constrained': {
            'position_dependent_weights': 1,
            'vector_size': 60,
            'window': 15,
        },
    },
    'build_vocab': {
        'trim_rule': None,
        'progress_per': 10000,
    },
    'build_vocab_keys': [
        'key_to_index',
        'index_to_key',
        'expandos',
        'cum_table',
        'corpus_count',
        'corpus_total_words',
    ],
    'train': {
        'compute_loss': False,
    },
    'train_keys': set([
        'epochs',
        'compute_loss',
    ]),
}

SIMPLE_PREPROCESS_PARAMETERS = {
    'deacc': False,
    'min_len': 0,
    'max_len': 15,
}


def MODEL_BASENAMES(subwords, positions):
    parts = []
    if positions == 'constrained':
        parts.append('constrained')
    if positions is not False:
        parts.append('positional')
    if subwords:
        parts.append('fasttext')
    else:
        parts.append('word2vec')
    parts.append('cbow')
    return '_'.join(parts)


def MODEL_FRIENDLY_NAMES(subwords, positions):
    if positions == 'constrained':
        return 'constrained'
    if positions == 'full':
        return 'positional'
    if subwords:
        return 'subword'
    return 'baseline'


WORD_ANALOGY_PARAMETERS = {
    'case_insensitive': True,
    'dummy4unknown': False,
    'restrict_vocab': 2 * 10**5,
}

WORD_ANALOGY_DATASETS = {
    'en': {
        'url': 'https://github.com/tmikolov/word2vec/raw/master/questions-words.txt',
        'size': 603955,
    },
}

TEXT_CLASSIFICATION_DATASETS = {
    'en': {
        'url': 'https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=1',
        'size': 4721151112,
    },
}

LANGUAGE_MODELING_DATASETS = {
    'url': 'https://github.com/bothameister/bothameister.github.io/raw/master/icml14-data.tar.bz2',
    'size': 52346880,
}

TEXT_CLASSIFICATION_METHOD_PARAMETERS = {
    'scm': {  # Parameters from paper <https://www.aclweb.org/anthology/S17-2051/>, Section 2.1
        'similarity_matrix': {
            'nonzero_limit': 100,
            'symmetric': True,
            'positive_definite': False,
        },
        'similarity_index': {
            'threshold': 0.0,
            'exponent': 2.0,
        },
    },
    'wmd': {
    }
}

TEXT_CLASSIFICATION_DATASET_SIZES = {
    'bbcsport': 517,
    'twitter': 2176,
    'recipe2': 3059,
    'ohsumed': 3999,
    'classic': 4965,
    'r8': 5485,
    'amazon': 5600,
    '20ng2_500': 11293,
}

LANGUAGE_MODELING_PARAMETERS = {
    'device': 'cuda',
    'tie_weights': True,
    'freeze_embeddings': True,
    'ninp': 300,
    'nhid': 300,
    'nlayers': 2,
    'dropout': 0.5,
    'batch_size': 40,
    'initial_learning_rate': 20,
    'eval_batch_size': 10,
    'sentence_length': 35,
    'annealing': 4.0,
    'clip': 0.25,
    'epochs': 50,
    'seed': 21,
}

FEATURE_CLUSTERING_PARAMETERS = {
    'linkage': 'ward',
    'affinity': 'euclidean',
}

NUM_FEATURE_CLUSTERS = {
    'full': 3,
    'constrained': 2,
}

FEATURE_CLUSTER_COLORS = defaultdict(
    lambda _: 4,
    {
        'antepositional': 0,
        'postpositional': 1,
        'informational': 2,
    },
)

CORPUS_SIZES = {
    'wikipedia': {
        'en': 249230825,
    },
    'common_crawl': {
        'en': 37070013808,
    },
}

PLOT_PARAMETERS = {
    'interpolation': {
        'kind': 'cubic',
        'num_points': 500,
    },
    'batch_smoothing': 1.0,
    'axis_gamma': 0.5,
    'line_alpha': 0.4,
    'line_gamma': 0.7,
    'language_modeling': {
        'kind': 'perplexity',
        'subset': 'validation',
    },
}

WORD_KINDS = [
    'context',
    'masked',
]

LANGUAGE_MODELING_RESULT_KINDS = [
    'loss',
    'perplexity',
    'learning_rate',
]

LANGUAGE_MODELING_RESULT_SUBSETS = [
    'train',
    'validation',
]

PICKLE_PROTOCOL = 3

JSON_DUMP_PARAMETERS = {
    'indent': 4,
    'sort_keys': True,
}

NUM_PRINTED_TOP_WORDS = 10
NUM_PRINTED_BOTTOM_WORDS = 5

EXAMPLE_SENTENCES = {
    'restrict_vocab': 10**3,
    'restrict_positions': (-3, 3),
    'whitelist': defaultdict(
        lambda _: None,
        {
            'en': set([
                *(word.lower() for word in nltk.corpus.brown.words()),
            ]),
        },
    ),
    'blacklist': defaultdict(
        lambda _: None,
        {
            'en': lambda word: word in ('a', 'an', 'the'),
        },
    ),
}
