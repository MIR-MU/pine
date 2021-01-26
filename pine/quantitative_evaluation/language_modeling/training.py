# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from logging import getLogger
from typing import Iterable, Union

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError('For training language models, please install PyTorch')

from tqdm import tqdm

from .language_modeling import Dataset, Result, TrainingResult, EvaluationResult
from .data import Corpus
from .model import PreinitializedRNNModel
from ...configuration import LANGUAGE_MODELING_PARAMETERS
from ...language_model import LanguageModel


LOGGER = getLogger(__name__)


def train_and_evaluate(dataset: Dataset, language_model: LanguageModel) -> Result:
    cache_path = language_model.cache_dir / 'language_modeling'
    cache_path = cache_path.with_suffix('.pt')

    torch.manual_seed(LANGUAGE_MODELING_PARAMETERS['seed'])
    device = torch.device(LANGUAGE_MODELING_PARAMETERS['device'])
    corpus = Corpus(dataset)
    model = PreinitializedRNNModel(language_model.vectors, dataset)
    model = model.to(device)
    corpus.dictionary = model.synchronize_mapping(corpus.dictionary)
    corpus.tokenize_all()

    def batchify(data: torch.Tensor, batch_size: int) -> torch.Tensor:
        nbatch = data.size(0) // batch_size
        data = data.narrow(0, 0, nbatch * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        return data.to(device)

    train_data = batchify(corpus.train, LANGUAGE_MODELING_PARAMETERS['batch_size'])
    validation_data = batchify(corpus.valid, LANGUAGE_MODELING_PARAMETERS['eval_batch_size'])
    test_data = batchify(corpus.test, LANGUAGE_MODELING_PARAMETERS['eval_batch_size'])
    criterion = nn.NLLLoss()

    def train() -> TrainingResult:
        model.train()
        hidden = model.init_hidden(LANGUAGE_MODELING_PARAMETERS['batch_size'])
        sentence_length = LANGUAGE_MODELING_PARAMETERS['sentence_length']
        training_result = []
        batches = range(0, train_data.size(0) - 1, sentence_length)
        batches = enumerate(batches)
        for batch, i in batches:
            data, targets = get_batch(train_data, i)
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = criterion(output, targets)
            loss.backward()

            clip = LANGUAGE_MODELING_PARAMETERS['clip']
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

            current_loss = loss.item()
            current_perplexity = math.exp(current_loss)
            training_result.append((current_perplexity, current_loss))
        return training_result

    def evaluate(data_source: torch.Tensor) -> EvaluationResult:
        model.eval()
        total_loss = 0.0
        hidden = model.init_hidden(LANGUAGE_MODELING_PARAMETERS['eval_batch_size'])
        with torch.no_grad():
            sentence_length = LANGUAGE_MODELING_PARAMETERS['sentence_length']
            for i in range(0, data_source.size(0) - 1, sentence_length):
                data, targets = get_batch(data_source, i)
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
                total_loss += len(data) * criterion(output, targets).item()

        current_loss = total_loss / (len(data_source) - 1)
        current_perplexity = math.exp(current_loss)
        evaluation_result = (current_perplexity, current_loss)
        return evaluation_result

    lr = LANGUAGE_MODELING_PARAMETERS['initial_learning_rate']
    best_validation_loss = None
    epoch_results = []
    epochs = range(1, LANGUAGE_MODELING_PARAMETERS['epochs'] + 1)
    epochs = tqdm(epochs, desc='Epoch')
    for epoch in epochs:
        training_result = train()
        validation_result = evaluate(validation_data)
        epoch_result = (training_result, validation_result, lr)
        epoch_results.append(epoch_result)

        _, validation_loss = validation_result
        if best_validation_loss is None or validation_loss < best_validation_loss:
            with cache_path.open('wb') as wf:
                torch.save(model, wf)
            best_validation_loss = validation_loss
        else:
            lr /= LANGUAGE_MODELING_PARAMETERS['annealing']

    with cache_path.open('rb') as rf:
        model = torch.load(rf)
        model.rnn.flatten_parameters()

    test_result = evaluate(test_data)
    test_perplexity, _ = test_result
    LOGGER.info('Test perplexity: {:.2f}'.format(test_perplexity))
    language_modeling_result = (test_result, epoch_results)
    return language_modeling_result


def repackage_hidden(h: NestedTensor) -> NestedTensor:
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


NestedTensor = Union[torch.Tensor, Iterable['NestedTensor']]


def get_batch(source: torch.Tensor, i: int) -> torch.Tensor:
    seq_len = min(LANGUAGE_MODELING_PARAMETERS['sentence_length'], len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
