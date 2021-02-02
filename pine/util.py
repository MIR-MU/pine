# -*- coding: utf-8 -*-

import re
import math
from numbers import Integral
from pathlib import Path
from logging import getLogger
from functools import partial
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import smart_open
from tqdm import tqdm
import tarfile
from zipfile import ZipFile
from scipy.interpolate import interp1d
import gensim.utils

from .configuration import PLOT_PARAMETERS, SIMPLE_PREPROCESS_PARAMETERS


LOGGER = getLogger(__name__)


def download_to(url: str, size: int, path: Path, buffer_size: int = 2**20):
    desc = 'Downloading {} to {}'.format(url, path)
    downloaded = 0
    with tqdm(total=size, unit='B', desc=desc) as pbar:
        with path.open('wb') as wf, smart_open.open(url, 'rb') as rf:
            for data in iter(partial(rf.read, buffer_size), b''):
                wf.write(data)
                downloaded += len(data)
                pbar.update(len(data))
    if size != downloaded:
        raise ValueError('Downloaded {} bytes, expected {} bytes'.format(downloaded, size))


def unzip_to(archive: Path, result_dir: Path, unlink_after: bool = False):
    LOGGER.info('Extracting {} to {}'.format(archive, result_dir))
    suffixes = set(archive.suffixes)
    if '.zip' in suffixes:
        with ZipFile(archive, 'r') as zf:
            zf.extractall(result_dir)
    elif '.tar' in suffixes:
        mode = 'r' if len(suffixes) == 1 else 'r:{}'.format(suffixes[-1])
        with tarfile.open(archive, mode) as tf:
            tf.extractall(result_dir)
    else:
        raise ValueError('Unsupported archive {}'.format(archive))

    if unlink_after:
        archive.unlink()


def stringify_parameters(parameters: Dict) -> str:
    def millify(n: Integral) -> str:
        millnames = ('', 'K', 'M', 'G', 'T')
        n = float(n)
        millidx = max(
            0,
            min(
                len(millnames) - 1,
                int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))
            )
        )
        return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

    def stringify(obj: Any) -> str:
        obj = millify(obj) if isinstance(obj, Integral) else str(obj)
        obj = re.sub('_', '-', obj)
        return obj

    parameters = sorted(parameters.items())
    parameters = ('{}={}'.format(stringify(key), stringify(value)) for key, value in parameters)
    return '_'.join(parameters)


def interpolate(X: np.ndarray, Y: np.ndarray, kind: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    parameters = PLOT_PARAMETERS['interpolation']
    if kind is None:
        kind = parameters['kind']
    interpolation_function = interp1d(X, Y, kind=kind)
    X = np.linspace(min(X), max(X), num=parameters['num_points'], endpoint=True)
    Y = interpolation_function(X)
    return (X, Y)


def simple_preprocess(document: str) -> List[str]:
    return gensim.utils.simple_preprocess(document, **SIMPLE_PREPROCESS_PARAMETERS)
