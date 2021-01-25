# -*- coding: utf-8 -*-

from pathlib import Path
from logging import getLogger
from functools import partial

import smart_open
from tqdm import tqdm
import tarfile
from zipfile import ZipFile


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
