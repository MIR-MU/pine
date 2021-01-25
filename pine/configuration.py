# -*- coding: utf-8 -*-

FASTTEXT_PARAMETERS = {
    'baseline': {
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
        },
        'full': {
            'position_dependent_weights': 1,
        },
        'constrained': {
            'position_dependent_weights': 1,
            'vector_size': 60,
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
    'url': 'https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=1',
    'size': 4721151112,
}

LANGUAGE_MODELING_DATASETS = {
    'url': 'https://github.com/bothameister/bothameister.github.io/raw/master/icml14-data.tar.bz2',
    'size': 10803893,
}

TEXT_CLASSIFICATION_METHODS = set([
    'scm',
    'wmd',
])

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

CORPUS_SIZES = {
    'wikipedia': {
        'en': 249230825,
    },
    'common_crawl': {
        'en': 37070013808,
    },
}

PICKLE_PROTOCOL = 3
