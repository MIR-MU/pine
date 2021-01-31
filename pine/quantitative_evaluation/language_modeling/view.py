# -*- coding: utf-8 -*-

from __future__ import annotations

from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ...util import interpolate
from ...language_model import LanguageModel
from ...configuration import PLOT_PARAMETERS, LANGUAGE_MODELING_RESULT_KINDS, LANGUAGE_MODELING_RESULT_SUBSETS


def plot_language_modeling_results(*language_models: LanguageModel,
                                   kind: str = None, subset: str = None) -> Figure:
    if kind is None:
        kind = PLOT_PARAMETERS['language_modeling']['kind']
    if kind not in LANGUAGE_MODELING_RESULT_KINDS:
        known_kinds = ', '.join(LANGUAGE_MODELING_RESULT_KINDS)
        message = 'Unknown kind {} (known kinds: {})'.format(kind, known_kinds)
        raise ValueError(message)

    if subset is None:
        subset = PLOT_PARAMETERS['language_modeling']['subset']
    if subset not in LANGUAGE_MODELING_RESULT_SUBSETS:
        known_subsets = ', '.join(LANGUAGE_MODELING_RESULT_SUBSETS)
        message = 'Unknown subset {} (known subsets: {})'.format(subset, known_subsets)
        raise ValueError(message)

    fig, ax = plt.subplots(1, 1)
    for language_model_number, language_model in enumerate(language_models):
        _, epoch_results = language_model.language_modeling.result
        X, Y = [], []
        last_epoch = len(epoch_results) - 1
        for epoch, epoch_result in enumerate(epoch_results):
            training_results, validation_result, learning_rate = epoch_result
            validation_perplexity, validation_loss = validation_result
            if subset == 'validation':
                X.append(epoch + 1)
                if kind == 'loss':
                    Y.append(validation_loss)
                elif kind == 'perplexity':
                    Y.append(validation_perplexity)
                else:
                    message = 'Plotting language modeling {} results of kind {} not yet implemented'
                    message = message.format(subset, kind)
                    raise ValueError(message)
            elif subset == 'train':
                if kind == 'learning_rate':
                    X.append(epoch)
                    Y.append(learning_rate)
                    if epoch == last_epoch:
                        X.append(epoch + 1)
                        Y.append(learning_rate)
                elif kind == 'loss' or kind == 'perplexity':
                    for batch, training_result in enumerate(training_results):
                        training_perplexity, training_loss = training_result
                        batch = epoch + float(batch) / len(training_results)
                        if batch % PLOT_PARAMETERS['batch_smoothing'] != 0:
                            continue
                        if kind == 'loss':
                            value = training_loss
                        elif kind == 'perplexity':
                            value = training_perplexity
                        X.append(batch)
                        Y.append(value)
                    if epoch == last_epoch:
                        X.append(epoch + 1)
                        Y.append(value)
                else:
                    message = 'Plotting language modeling {} results of kind {} not yet implemented'
                    message = message.format(subset, kind)
                    raise ValueError(message)
            else:
                message = 'Plotting language modeling results for subset {} not yet implemented'
                message = message.format(subset)
                raise ValueError(message)
        interpolation_kind = 'previous' if kind == 'learning_rate' else None
        X, Y = interpolate(X, Y, interpolation_kind)
        ax.plot(X, Y, label=language_model.friendly_name.capitalize())
    ax.legend()
    return fig
