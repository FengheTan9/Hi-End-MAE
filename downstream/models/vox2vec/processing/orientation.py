import numpy as np
from connectome import Transform, positional, optional, inverse, Input


class FlipAxesToCanonical(Transform):
    __inherit__ = True

    @positional
    def image(x, flipped_axes):
        if not flipped_axes:
            return x
        return np.flip(x, flipped_axes).copy()

    @optional
    @positional
    def mask(x, flipped_axes):
        if not flipped_axes or x is None:
            return x
        return np.flip(x, flipped_axes).copy()

    def flipped_axes(flipped_axes):
        return ()

    @inverse
    def sgm(sgm, flipped_axes: Input):
        if not flipped_axes:
            return sgm
        return np.flip(sgm, flipped_axes).copy()
