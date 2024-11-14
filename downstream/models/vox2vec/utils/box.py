from typing import *
import itertools
import numpy as np

from scipy.optimize import linear_sum_assignment


def mask_to_bbox(mask: np.ndarray) -> np.ndarray:
    """
    Find the smallest box that contains all true values of the ``mask``.
    """
    if not mask.any():
        raise ValueError('The mask is empty.')

    start, stop = [], []
    for ax in itertools.combinations(range(mask.ndim), mask.ndim - 1):
        nonzero = np.any(mask, axis=ax)
        if np.any(nonzero):
            left, right = np.where(nonzero)[0][[0, -1]]
        else:
            left, right = 0, 0
        start.insert(0, left)
        stop.insert(0, right + 1)

    return np.array([start, stop])


def limit_box(box: np.ndarray, limit: Union[int, Sequence[int]]) -> np.ndarray:
    start, stop = box
    start = np.maximum(start, 0)
    stop = np.minimum(stop, limit)
    return np.array([start, stop])
