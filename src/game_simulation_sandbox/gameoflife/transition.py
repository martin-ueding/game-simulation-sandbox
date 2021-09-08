from typing import *

import numpy as np


def add_padding(state: np.ndarray) -> np.ndarray:
    return np.pad(state, 1, "constant")
