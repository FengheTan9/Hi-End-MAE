from typing import *
from pathlib import Path
from functools import partial
import numpy as np
import nibabel
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from connectome import Source, meta, Chain, Transform, Apply, CacheToRam, CacheToDisk

from vox2vec.utils.box import mask_to_bbox
from vox2vec.processing import (
    RescaleToSpacing, FlipAxesToCanonical, scale_hu, CropToBox, LabelsToOnehot,
    sample_patches, get_body_mask, BODY_THRESHOLD_HU
)
from vox2vec.utils.split import kfold
from vox2vec.utils.data import VanillaDataset, ResizeByRandomSampling


LABELS = {
    1: 'Spleen',
    2: 'Right kidney',
    3: 'Left kidney',
    4: 'Gallbladder',
    5: 'Esophagus',
    6: 'Liver',
    7: 'Stomach',
    8: 'Aorta',
    9: 'Inferior Vena Cava (IVC)',
    10: 'Portal Vein and Splenic Vein',
    11: 'Pancreas',
    12: 'Right Adrenal Gland',
    13: 'Left Adrenal Gland',
}


class BTCVSource(Source):
    _root: str

    @meta
    def train_ids(_root):
        return sorted({
            file.name[len('img'):-len('.nii.gz')]
            for file in Path(_root).glob('**/Training/img/img*.nii.gz')
        })

    @meta
    def test_ids(_root):
        return sorted({
            file.name[len('img'):-len('.nii.gz')]
            for file in Path(_root).glob('**/Testing/img/img*.nii.gz')
        })

    def _image_nii(id_, _root):
        file,  = Path(_root).glob(f'**/img{id_}.nii.gz')
        return nibabel.load(file)

    def _mask_nii(id_, _root):
        try:
            file, = Path(_root).glob(f'**/label{id_}.nii.gz')
        except ValueError:
            return

        return nibabel.load(file)

    def image(_image_nii):
        return _image_nii.get_fdata().astype(np.float32)

    def affine(_image_nii):
        return _image_nii.affine

    def mask(_mask_nii):
        if _mask_nii is not None:
            return _mask_nii.get_fdata().astype(np.int16)


class BTCV(pl.LightningDataModule):
    num_classes = len(LABELS)

    def __init__(
            self,
            root: str,
            cache_dir: str,
            spacing: Tuple[float, float, float],
            window_hu: Tuple[float, float],
            patch_size: Tuple[int, int, int],
            batch_size: int,
            num_batches_per_epoch: Optional[int],
            num_workers: int,
            prefetch_factor: int,
            split: int,
            num_splits: int = 5,
            val_size: Union[float, int] = 1,
            train_size: Optional[Union[float, int]] = None,
            cache_to_ram: bool = True,
            random_state: int = 42
    ) -> None:
        super().__init__()

        source = BTCVSource(root=root)

        train_val_test_ids = source.train_ids
        pred_ids = source.test_ids
        split_gen = kfold(train_val_test_ids, num_splits, random_state)
        train_val_ids, test_ids = list(split_gen)[split]
        train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size, train_size=train_size,
                                              random_state=random_state)

        # use connectome for smart cashing (with automatic invalidation)
        preprocessing = Chain(
            Transform(
                __inherit__=True,
                flipped_axes=lambda affine: tuple(np.where(np.diag(affine[:3, :3]) < 0)[0] - 3),  # enumerate from the end
                spacing=lambda affine: tuple(np.abs(np.diag(affine[:3, :3]))),
            ),
            FlipAxesToCanonical(),
            Transform(__inherit__=True, cropping_box=lambda image: mask_to_bbox(image >= BODY_THRESHOLD_HU)),
            CropToBox(axis=(-3, -2, -1)),
            RescaleToSpacing(to_spacing=spacing, axis=(-3, -2, -1), image_fill_value=np.min),
            Transform(__inherit__=True, body_mask=lambda image: get_body_mask(image)),
            Apply(image=partial(scale_hu, window_hu=window_hu))
        )

        train_pipeline = source >> preprocessing >> CacheToDisk.simple('image', 'body_mask', 'mask', root=cache_dir)
        if cache_to_ram:
            train_pipeline >>= CacheToRam(['image', 'body_mask', 'mask'])
        train_pipeline >>= LabelsToOnehot(labels_range=range(1, len(LABELS) + 1))
        train_pipeline >>= Apply(image=lambda x: x[None], mask=np.float32)

        _load_train_example = train_pipeline._compile(['image', 'body_mask', 'mask'])

        def load_train_example(id_):
            examples = sample_patches(*_load_train_example(id_), patch_size, batch_size)
            return [torch.tensor(np.stack(xs)) for xs in zip(*examples)]

        load_val_example = load_test_example = _load_train_example
        load_pred_example = source._compile(['image', 'affine'])

        self.train_dataset = ResizeByRandomSampling(VanillaDataset(train_ids, load_train_example), num_batches_per_epoch)
        self.val_dataset = VanillaDataset(val_ids, load_val_example)
        self.test_dataset = VanillaDataset(test_ids, load_test_example)
        self.pred_dataset = VanillaDataset(pred_ids, load_pred_example)
        self.pred_decorator = preprocessing._decorate(['image', 'body_mask'], 'sgm')
        
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=None)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=None)
