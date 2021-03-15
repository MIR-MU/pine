# -*- coding: utf-8 -*-

from collections import defaultdict
from pathlib import Path
import re

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
        'workers': 8,
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

CORPORA = {
    'common_crawl': {
        'en': [
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.00.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.01.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.02.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.03.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.04.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.05.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.06.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.07.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.08.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.09.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.10.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.11.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.12.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.13.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.14.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.15.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.16.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.17.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.18.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.19.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.20.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.21.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.22.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.23.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.24.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.25.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.26.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.27.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.28.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.29.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.30.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.31.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.32.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.33.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.34.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.35.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.36.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.37.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.38.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.39.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.40.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.41.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.42.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.43.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.44.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.45.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.46.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.47.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.48.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.49.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.50.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.51.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.52.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.53.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.54.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.55.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.56.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.57.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.58.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.59.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.60.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.61.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.62.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.63.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.64.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.65.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.66.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.67.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.68.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.69.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.70.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.71.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.72.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.73.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.74.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.75.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.76.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.77.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.78.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.79.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.80.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.81.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.82.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.83.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.84.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.85.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.86.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.87.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.88.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.89.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.90.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.91.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.92.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.93.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.94.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.95.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.96.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.97.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.98.deduped.xz',
            },
            {
                'url': 'http://web-language-models.s3-website-us-east-1.amazonaws.com/ngrams/en/deduped/en.99.deduped.xz',
            },
        ],
    },
}

WORD_ANALOGY_DATASETS = {
    'cs': {
        'url': 'https://raw.githubusercontent.com/Svobikl/cz_corpus/master/corpus/czech_emb_corpus_no_phrase.txt',
        'size': 712906,
    },
    'de': {
        'url': 'https://www.ims.uni-stuttgart.de/documents/ressourcen/lexika/analogies_ims/analogies.zip',
        'size': 250507,
        'extract_file': Path('analogies') / 'de_trans_Google_analogies.txt',
    },
    'en': {
        'url': 'https://github.com/tmikolov/word2vec/raw/master/questions-words.txt',
        'size': 603955,
    },
    'es': {
        'url': 'https://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/questions-words_sp.txt',
        'size': 490314,
    },
    'fi': {
        'url': 'https://github.com/Witiko/FinSemEvl/releases/download/v1.0/fi.txt',
        'size': 30439,
    },
    'fr': {
        'url': 'https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-fr.txt',
        'size': 1073616,
        'transformation': lambda line: re.sub(r'^\s*:', ':', line)
    },
    'hi': {
        'url': 'https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-hi.txt',
        'size': 2213485,
        'transformation': lambda line: re.sub(r'^\s*:', ':', line)
    },
    'pl': {
        'url': 'https://dl.fbaipublicfiles.com/fasttext/word-analogies/questions-words-pl.txt',
        'size': 825883,
        'transformation': lambda line: re.sub(r'^\s*:', ':', line)
    },
    'pt': {
        'url': 'https://raw.githubusercontent.com/nathanshartmann/portuguese_word_embeddings/master/analogies/testset/LX-4WAnalogies.txt',
        'size': 598795,
    },
    'tr': {
        'url': 'https://github.com/Witiko/linguistic-features-in-turkish-word-representations/releases/download/v1.0/tr.txt',
        'size': 1944843,
    },
    'zh': {
        'url': 'https://raw.githubusercontent.com/Leonard-Xu/CWE/master/data/analogy.txt',
        'size': 30585,
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
    lambda: 4,
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
        lambda: None,
        {
            'en': set([
                *(word.lower() for word in nltk.corpus.brown.words()),
            ]),
        },
    ),
    'blacklist': defaultdict(
        lambda: None,
        {
            'en': lambda word: word in ('a', 'an', 'the'),
        },
    ),
}
