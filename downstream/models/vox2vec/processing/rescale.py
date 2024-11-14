from typing import *
import numpy as np
import warnings

from connectome import Transform, Mixin, optional, inverse, Input, positional
from imops import zoom, zoom_to_shape

from vox2vec.utils.misc import normalize_axis_list


class _Rescale(Mixin):
    def image(image, _scale_factor, _axis, _image_fill_value):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return zoom(image, _scale_factor, _axis, fill_value=_image_fill_value, backend='Scipy')

    @optional
    def spacing(spacing, _scale_factor, _axis):
        axis = normalize_axis_list(_axis, len(spacing))
        spacing = np.array(spacing)
        spacing[axis] /= _scale_factor
        return tuple(spacing)

    @optional
    def shape(shape, _scale_factor, _axis):
        axis = normalize_axis_list(_axis, len(shape))
        shape = np.array(shape)
        shape[axis] = np.round(shape[axis] * _scale_factor).astype(int)
        return tuple(shape)

    @optional
    @positional
    def mask(x, _scale_factor, _axis):
        if x is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return zoom(x, _scale_factor, _axis, order=0, backend='Scipy')

    body_mask = mask

    @inverse
    def sgm(sgm, image: Input, _axis, _sgm_interp_ord):
        shape = np.array(image.shape)[normalize_axis_list(_axis, image.ndim)]
        return zoom_to_shape(sgm, shape, _axis, order=_sgm_interp_ord, backend='Scipy')


class RescaleToSpacing(Transform, _Rescale):
    __inherit__ = 'id'
    _to_spacing: Union[float, Sequence[float]]
    _axis: Union[int, Sequence[int]]
    _image_fill_value: Union[float, Callable]
    _sgm_interp_ord: int = 1

    def _scale_factor(spacing, _to_spacing, _axis):
        axis = normalize_axis_list(_axis, len(spacing))
        old_spacing = np.array(spacing, dtype=float)[axis]
        return np.nan_to_num(old_spacing / _to_spacing, nan=1)


class RescaleToShape(Transform, _Rescale):
    __inherit__ = 'id'
    _shape: Union[int, Sequence[int]]
    _axis: Union[int, Sequence[int]]
    _image_fill_value: Union[float, Callable]
    _sgm_interp_ord: int = 1

    def _scale_factor(image, _shape, _axis):
        axis = normalize_axis_list(_axis, image.ndim)
        old_shape = np.array(image.shape, dtype=float)[axis]
        return _shape / old_shape


def locations_to_spacing(locations, q=0.95, raise_error=True):
    spacings, counts = np.unique(np.round(np.abs(np.diff(locations)), 2), return_counts=True)
    if counts.max() >= len(locations) * q:
        return spacings[counts.argmax()]
    elif raise_error:
        raise ValueError('Non-uniform locations.')


class LocationsToSpacing(Transform):
    __inherit__ = True

    def spacing(pixel_spacing, slice_locations):
        return (*pixel_spacing, locations_to_spacing(slice_locations, raise_error=False))
