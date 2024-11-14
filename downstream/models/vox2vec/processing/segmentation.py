from typing import Iterable
import numpy as np

from connectome import Transform


def labels_to_onehot(labels: np.ndarray, labels_range: Iterable[int]):
    return np.stack([labels == l for l in labels_range])


def onehot_to_labels(mask: np.ndarray, threshold: float = 0.5):
    mask = np.insert(mask, 0, threshold, axis=0)
    return np.argmax(mask, axis=0)


class LabelsToOnehot(Transform):
    __inherit__ = True
    _labels_range: Iterable

    def mask(_labels_range, mask):
        if mask is not None:
            return labels_to_onehot(mask, _labels_range)
