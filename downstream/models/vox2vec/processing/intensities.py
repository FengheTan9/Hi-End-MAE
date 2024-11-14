from typing import *
import numpy as np
from scipy.ndimage import gaussian_filter1d

from vox2vec.utils.misc import normalize_axis_list


def scale_hu(image_hu: np.ndarray, window_hu: Tuple[float, float]) -> np.ndarray:
    min_hu, max_hu = window_hu
    assert min_hu < max_hu
    return np.clip((image_hu - min_hu) / (max_hu - min_hu), 0, 1)


def gaussian_filter(
        x: np.ndarray,
        sigma: Union[float, Sequence[float]],
        axis: Union[int, Sequence[int]]
) -> np.ndarray:
    axis = normalize_axis_list(axis, x.ndim)
    sigma = np.broadcast_to(sigma, len(axis))
    for sgm, ax in zip(sigma, axis):
        x = gaussian_filter1d(x, sgm, ax)
    return x


def gaussian_sharpen(
        x: np.ndarray,
        sigma_1: Union[float, Sequence[float]],
        sigma_2: Union[float, Sequence[float]],
        alpha: float,
        axis: Union[int, Sequence[int]]
) -> np.ndarray:
    """ See https://docs.monai.io/en/stable/transforms.html#gaussiansharpen """
    blurred = gaussian_filter(x, sigma_1, axis)
    return blurred + alpha * (blurred - gaussian_filter(blurred, sigma_2, axis))
