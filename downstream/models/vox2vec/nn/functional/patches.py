from typing import *
import numpy as np
import torch

from models.vox2vec.utils.misc import collect


@collect
def yield_sw_boxes(input_size, patch_size, overlap):
    stride = patch_size - overlap
    for start in np.ndindex(tuple(1 + np.int64(np.ceil((input_size - patch_size) / stride)))):
        start *= stride
        stop = np.minimum(start + patch_size, input_size)
        start = stop - patch_size
        yield np.array([start, stop])


def sw_predict(
        images: torch.Tensor,
        predict: callable,
        patch_size: Tuple[int, int, int],
        overlap: float = 0.5,
) -> torch.Tensor:
    (batch_size, _, *image_size) = images.shape
    image_size = np.array(image_size)
    patch_size = np.array(patch_size)
    overlap = np.int64(np.ceil(patch_size * overlap))
    overlap += overlap % 2
    assert np.all(overlap % 2 == 0)

    for i, box in enumerate(yield_sw_boxes(image_size, patch_size, overlap)):
        patches = images[(..., *map(slice, *box))]
        patches_sgm = predict(patches)

        if i == 0:
            assert patches_sgm.ndim == 5
            num_classes = patches_sgm.shape[1]
            sgm = torch.zeros((batch_size, num_classes, *image_size)).to(images)

        start, stop = box
        start = np.where(start > 0, start + overlap // 2, 0)
        stop = np.where(stop < image_size, stop - overlap // 2, image_size)
        sgm[(..., *map(slice, start, stop))] = patches_sgm[(..., *map(slice, start - box[0], stop - box[0]))]

    return sgm
