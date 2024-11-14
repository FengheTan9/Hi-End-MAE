from typing import *
import numpy as np
import random

from imops import crop_to_box

from vox2vec.utils.misc import collect


def sample_box(image_size, patch_size, anchor_voxel=None):
    image_size = np.array(image_size, ndmin=1)
    patch_size = np.array(patch_size, ndmin=1)

    if not np.all(image_size >= patch_size):
        raise ValueError(f'Can\'t sample patch of size {patch_size} from image of size {image_size}')

    min_start = 0
    max_start = image_size - patch_size
    if anchor_voxel is not None:
        anchor_voxel = np.array(anchor_voxel, ndmin=1)
        min_start = np.maximum(min_start, anchor_voxel - patch_size + 1)
        max_start = np.minimum(max_start, anchor_voxel)
    start = np.random.randint(min_start, max_start + 1)
    return np.array([start, start + patch_size])


@collect
def sample_patches(
        image: np.ndarray,
        roi: np.ndarray,
        mask: np.ndarray,
        patch_size: Tuple[int, int, int],
        num_patches: int,
        resampling_p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_size = np.array(image.shape[-3:])
    assert np.all(patch_size <= image_size)

    assert mask.ndim == 4
    resampling_voxels = np.argwhere(random.choice(mask))
    for _ in range(num_patches):
        if len(resampling_voxels) > 0 and random.uniform(0, 1) < resampling_p:
            box = sample_box(image_size, patch_size, anchor_voxel=random.choice(resampling_voxels))
        else:
            box = sample_box(image_size, patch_size)

        yield (
            crop_to_box(image, box, axis=(-3, -2, -1)),
            crop_to_box(roi, box),
            crop_to_box(mask, box, axis=(-3, -2, -1))
        )
