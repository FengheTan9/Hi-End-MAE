from typing import *
import json
import more_itertools
from functools import wraps
from joblib import Parallel, delayed, parallel_backend
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
import torch


def normalize_axis_list(axis, ndim):
    return list(normalize_axis_tuple(axis, ndim))


def collect(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))

    return wrapper


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.data.cpu().numpy()


def is_diagonal(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, np.diag(np.diag(matrix)))


def apply_along_axes(func, x, axis, n_jobs=-1, *args, **kwargs):
    """
    Apply ``func`` to slices from ``x`` taken along ``axes``.
    ``args`` and ``kwargs`` are passed as additional arguments.

    Notes
    -----
    ``func`` must return an array of the same shape as it received.
    """
    axis = normalize_axis_tuple(axis, x.ndim)
    if len(axis) == x.ndim:
        return func(x)

    last = tuple(range(-len(axis), 0))
    x = np.moveaxis(x, axis, last)

    with parallel_backend('threading', n_jobs=n_jobs):
        results = Parallel()(
            delayed(func)(patch, *args, **kwargs) for patch in x.reshape(-1, *x.shape[-len(axis):]))

    result = np.stack(results)
    return np.moveaxis(result.reshape(*x.shape), last, axis)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f)


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
