# -*- coding: utf-8 -*-

from pathlib import Path
from functools import partial

import smart_open
from tqdm import tqdm


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
