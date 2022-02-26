from typing import *

import numpy as np
import scipy.signal


def next_state(state: np.ndarray) -> np.ndarray:
    state = add_padding(state)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbors = scipy.signal.convolve2d(state, kernel, mode="same")
    assert np.all(state.shape == neighbors.shape)
    new_state = 1 * ((state == 1) & ((neighbors == 2) | (neighbors == 3))) | (
        (state == 0) & (neighbors == 3)
    )
    return remove_padding(new_state)


def add_padding(state: np.ndarray) -> np.ndarray:
    return np.pad(state, 1, "constant")


def remove_padding(state: np.ndarray) -> np.ndarray:
    for min1 in range(state.shape[0]):
        if np.any(state[min1, :] != 0):
            break
    for max1 in range(state.shape[0] - 1, -1, -1):
        if np.any(state[max1, :] != 0):
            break
    for min2 in range(state.shape[1]):
        if np.any(state[:, min2] != 0):
            break
    for max2 in range(state.shape[1] - 1, -1, -1):
        if np.any(state[:, max2] != 0):
            break
    return state[min1 : max1 + 1, min2 : max2 + 1]
