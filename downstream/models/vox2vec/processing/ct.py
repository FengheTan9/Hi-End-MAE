from typing import *
import numpy as np
import math
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import flood
from imops import pad

from vox2vec.utils.misc import apply_along_axes


BODY_THRESHOLD_HU = -500


def get_body_mask(image: np.ndarray) -> np.ndarray:
    air = image < BODY_THRESHOLD_HU
    return ~flood(pad(air, padding=1, axis=(0, 1), padding_values=True), seed_point=(0, 0, 0))[1:-1, 1:-1]
