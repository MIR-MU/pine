# -*- coding: utf-8 -*-

from ..language_model import LanguageModel


def position_to_index(language_model: LanguageModel, position: int) -> int:
    if position == 0:
        message = 'Position {} is not a valid position of a context word'
        message = message.format(position)
        raise ValueError(message)
    min_position = index_to_position(language_model, 0)
    max_position = index_to_position(language_model, len(language_model.positional_vectors) - 1)
    if position < min_position or position > max_position:
        message = 'Position {} is outside the interval [{}; {}]'
        message = message.format(position, min_position, max_position)
        raise ValueError(message)
    window_center = len(language_model.positional_vectors) // 2
    index = position + window_center
    if position > 0:
        index -= 1
    return index


def index_to_position(language_model: LanguageModel, index: int) -> int:
    min_index, max_index = 0, len(language_model.positional_vectors) - 1
    if index < min_index or index > max_index:
        message = 'Position index {} is outside the interval [{}; {}]'
        message = message.format(index, min_index, max_index)
        raise ValueError(message)
    window_center = len(language_model.positional_vectors) // 2
    position = index - window_center
    if position >= 0:
        position += 1
    return position
