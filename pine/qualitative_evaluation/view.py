# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Tuple

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from ..configuration import PLOT_PARAMETERS
from ..language_model import LanguageModel


def get_position_numbers(language_model: LanguageModel) -> np.ndarray:
    position_numbers = np.arange(language_model.model.window * 2)
    position_numbers -= len(language_model.positional_vectors) // 2
    position_numbers[position_numbers >= 0] += 1
    return position_numbers


def interpolate(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    parameters = PLOT_PARAMETERS['interpolation']
    interpolation_function = interp1d(X, Y, kind=parameters['kind'])
    X = np.linspace(min(X), max(X), num=parameters['num_points'], endpoint=True)
    Y = interpolation_function(X)
    return (X, Y)


def gamma_forward(x):
    return np.sign(x) * np.abs(x)**PLOT_PARAMETERS['gamma']


def gamma_inverse(x):
    return np.sign(x) * np.abs(x)**(1.0 / PLOT_PARAMETERS['gamma'])


def plot_relative_importance_of_positions(*language_models: LanguageModel) -> Figure:
    if len(language_models) < 1:
        message = 'Expected at least one language model, got {}'
        message = message.format(len(language_models))
        raise ValueError(message)
    fig, ax = plt.subplots(1, 1)
    for language_model in language_models:
        relative_importance = language_model.relative_position_importance
        X = get_position_numbers(language_model)
        Y = relative_importance.data
        ax.scatter(X, Y, zorder=2)
        X, Y = interpolate(X, gamma_forward(Y))
        label = language_model.friendly_name
        ax.plot(X, gamma_inverse(Y), label=label.capitalize(), zorder=3)
    ax.grid(True, axis='both')
    ax.legend()
    ax.set_yscale('function', functions=(gamma_forward, gamma_inverse))
    return fig
