# -*- coding: utf-8 -*-

from __future__ import annotations

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from ..configuration import PLOT_PARAMETERS, FEATURE_CLUSTER_COLORS
from ..language_model import LanguageModel
from ..util import interpolate


def get_position_numbers(language_model: LanguageModel) -> np.ndarray:
    position_numbers = np.arange(len(language_model.positional_vectors))
    position_numbers -= len(language_model.positional_vectors) // 2
    position_numbers[position_numbers >= 0] += 1
    return position_numbers


def gamma_forward(x):
    return np.sign(x) * np.abs(x)**PLOT_PARAMETERS['axis_gamma']


def gamma_inverse(x):
    return np.sign(x) * np.abs(x)**(1.0 / PLOT_PARAMETERS['axis_gamma'])


def plot_position_importance(*language_models: LanguageModel) -> Figure:
    if len(language_models) < 1:
        message = 'Expected at least one language model, got {}'
        message = message.format(len(language_models))
        raise ValueError(message)
    fig, ax = plt.subplots(1, 1)
    for language_model in language_models:
        importance = language_model.position_importance
        X = get_position_numbers(language_model)
        Y = importance.data
        ax.scatter(X, Y, zorder=2)
        X, Y = interpolate(X, gamma_forward(Y))
        label = language_model.friendly_name
        ax.plot(X, gamma_inverse(Y), label=label.capitalize(), zorder=3)
    ax.legend()
    ax.set_yscale('function', functions=(gamma_forward, gamma_inverse))
    return fig


def plot_clustered_positional_features(language_model: LanguageModel) -> Figure:
    X = get_position_numbers(language_model)
    absolute_positional_features = np.abs(language_model.positional_vectors).T
    positional_feature_clusters = language_model.positional_feature_clusters
    labels = [
        *[
            label
            for label
            in FEATURE_CLUSTER_COLORS
            if label in positional_feature_clusters
        ],
        *sorted([
            label
            for label, _
            in positional_feature_clusters
            if label not in FEATURE_CLUSTER_COLORS
        ]),
    ]
    cmap = plt.get_cmap()
    fig, ax = plt.subplots(1, 1)
    for label in labels:
        color = cmap(FEATURE_CLUSTER_COLORS[label])
        indexes = positional_feature_clusters[label]
        cluster_size = len(indexes)
        linewidth = plt.rcParams['lines.linewidth'] / cluster_size**PLOT_PARAMETERS['line_gamma']
        for Y in absolute_positional_features[indexes]:
            ax.plot(*interpolate(X, Y), color=color, linewidth=linewidth)
        Y = np.mean(absolute_positional_features[indexes], axis=0)
        label = '{} features ({})'.format(label, cluster_size)
        ax.scatter(X, Y, color=color, zorder=2)
        ax.plot(*interpolate(X, Y), label=label.capitalize(), color=color, zorder=3)
    ax.legend()
    return fig
