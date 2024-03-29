# -*- coding: utf-8 -*-

import re
import math
from lzma import LZMAFile, FORMAT_XZ
from numbers import Integral
from pathlib import Path
from logging import getLogger
from functools import partial
from typing import Dict, Any, Tuple, Optional, List, Callable, TypeVar, Iterable
from tempfile import TemporaryFile, TemporaryDirectory
from shutil import copyfile

import numpy as np
import smart_open
from tqdm import tqdm
import tarfile
from zipfile import ZipFile
from scipy.interpolate import interp1d
import gensim.utils

from .configuration import PLOT_PARAMETERS, SIMPLE_PREPROCESS_PARAMETERS, SIMPLE_PREPROCESS_CHUNKSIZE


LOGGER = getLogger(__name__)


def download_to(url: str, path: Path, size: Optional[int] = None,
                transformation: Optional[Callable[[str], str]] = None,
                extract_file: Optional[Path] = None,
                buffer_size: int = 2**20):
    desc = 'Downloading {} to {}'.format(url, path)
    downloaded = 0
    with tqdm(total=size, unit='B', desc=desc) as pbar:
        with path.open('wb') as wf, smart_open.open(url, 'rb') as rf:
            for data in iter(partial(rf.read, buffer_size), b''):
                wf.write(data)
                downloaded += len(data)
                pbar.update(len(data))
    if size is not None and size != downloaded:
        raise ValueError('Downloaded {} bytes, expected {} bytes'.format(downloaded, size))
    if extract_file is not None:
        with TemporaryDirectory() as dirname:
            dirname = Path(dirname)
            unzip_to(path, dirname)
            copyfile(dirname / extract_file, path)
    if transformation is not None:
        rwf = TemporaryFile('w+t')
        with path.open('rt') as rf:
            for line in rf:
                line = line.rstrip('\r\n')
                line = transformation(line)
                print(line, file=rwf)
        rwf.seek(0)
        with path.open('wt') as wf:
            for line in rwf:
                line = line.rstrip('\r\n')
                print(line, file=wf)
        rwf.close()


def unzip_to(archive: Path, result_dir: Path, unlink_after: bool = False):
    LOGGER.info('Extracting {} to {}'.format(archive, result_dir))
    suffixes = set(archive.suffixes)
    if '.zip' in suffixes:
        with ZipFile(archive, 'r') as zf:
            zf.extractall(result_dir)
    elif '.tar' in suffixes:
        mode = 'r' if len(suffixes) == 1 else 'r:{}'.format(archive.suffixes[-1])
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


T = TypeVar('T')


def produce(iterable: Iterable[T], semaphore) -> Iterable[T]:
    for item in iterable:
        semaphore.acquire()
        yield item


def parallel_simple_preprocess(pool, path: Path, semaphore) -> Iterable[List[str]]:
    with path.open('rt') as f:
        producer = produce(f, semaphore)
        iterable = pool.imap(simple_preprocess, producer, SIMPLE_PREPROCESS_CHUNKSIZE)
        for line in iterable:
            yield line


def _handle_xz(file_obj, mode):
    return LZMAFile(filename=file_obj, mode=mode, format=FORMAT_XZ)


smart_open.register_compressor('.xz', _handle_xz)
