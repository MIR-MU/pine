# -*- coding: utf-8 -*-

from __future__ import annotations

from logging import getLogger
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError('For training language models, please install PyTorch')

import numpy as np
from gensim.models import KeyedVectors

from .language_modeling import Dataset
from .data import read_text_file, Dictionary
from ...configuration import LANGUAGE_MODELING_PARAMETERS


LOGGER = getLogger(__name__)


class RNNModel(nn.Module):
    def __init__(self, ntoken: int):
        super(RNNModel, self).__init__()

        ninp = LANGUAGE_MODELING_PARAMETERS['ninp']
        nhid = LANGUAGE_MODELING_PARAMETERS['nhid']
        nlayers = LANGUAGE_MODELING_PARAMETERS['nlayers']
        dropout = LANGUAGE_MODELING_PARAMETERS['dropout']
        tie_weights = LANGUAGE_MODELING_PARAMETERS['tie_weights']

        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = getattr(nn, 'LSTM')(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


class PreinitializedRNNModel(RNNModel):
    def __init__(self, vectors: KeyedVectors, dataset: Dataset):
        ninp = LANGUAGE_MODELING_PARAMETERS['ninp']
        freeze = LANGUAGE_MODELING_PARAMETERS['freeze_embeddings']

        ndims = vectors.vectors.shape[1]

        if ninp > ndims:
            message = 'Can\'t create a lookup table with {} dims from word vectors with {} dims'
            message = message.format(ninp, ndims)
            raise ValueError(message)
        if ninp < ndims:
            message = 'Trimming word vectors to the first {} dimensions out of {}'
            message = message.format(ninp, ndims)
            LOGGER.warn(message)

        vocab_words = ['<eos>']
        vocab_vecs = [vectors.vectors.mean(axis=0)[:ninp]]
        for word in read_text_file(dataset['vocab']):
            if word not in vectors:
                continue
            vocab_words.append(word)
            vocab_vecs.append(vectors[word][:ninp])

        super(PreinitializedRNNModel, self).__init__(len(vocab_words))

        weights = torch.FloatTensor(np.vstack(vocab_vecs))
        self.encoder = nn.Embedding.from_pretrained(weights, freeze=freeze)
        self.word2idx = {w: i for i, w in enumerate(vocab_words)}
        self.idx2word = dict(enumerate(vocab_words))
        self.ntoken = len(vocab_words)

    def synchronize_mapping(self, dictionary: Dictionary) -> Dictionary:
        dictionary.word2idx = self.word2idx
        dictionary.idx2word = self.idx2word
        return dictionary
