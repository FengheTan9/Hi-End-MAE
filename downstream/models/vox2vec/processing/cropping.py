from typing import *
import numpy as np

from connectome import Transform, Mixin, positional, inverse, Input, optional
from imops import crop_to_box, pad

from vox2vec.utils.box import limit_box
from vox2vec.utils.misc import normalize_axis_list, collect


def restore_crop(x, box, shape, axis, padding_values=0):
    axis = normalize_axis_list(axis, x.ndim)

    assert box.shape[1] == len(shape) == len(axis)

    padding = np.array([box[0], shape - box[1]], dtype=int).T
    return pad(x, padding, axis, padding_values)


class _CropToBox(Mixin):
    @positional
    def image(x, _box, _axis):
        if x is not None:
            return crop_to_box(x, _box, _axis)

    mask = body_mask = optional(image)

    def shape(_box):
        return tuple(_box[1] - _box[0])

    @inverse
    def sgm(sgm, image: Input, _box, _axis):
        shape = np.array(image.shape)[normalize_axis_list(_axis, image.ndim)]
        return restore_crop(sgm, _box, shape, _axis)


class CropToBox(Transform, _CropToBox):
    __inherit__ = 'spacing', 'id'
    _axis: Union[Sequence[int], int]

    def _box(cropping_box):
        return cropping_box
