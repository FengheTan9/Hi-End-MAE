import os
import torch
from monai.data.dataset import Dataset
from monai import transforms
import random
import numpy as np

class LargeMedicalDataSets(Dataset):

    def __init__(self, args):
    
        self.sample_list = []
        ls = []
        for op in os.listdir(args.base_dir1):
            name = op.split("_")
            name = name[0] + "_" + name[1]
            if name not in ["Dataset024_WORD", "Dataset009_AMOS"]:
                self.sample_list.append(os.path.join(args.base_dir1, op))
                ls.append(name)

        arr = np.array(ls)
        v, c = np.unique(arr, return_counts=True)
        for v, c in zip(v, c):
            print(v, c)
        # Shuffle the list in place
        random.shuffle(self.sample_list)

        # Slice the shuffled list
        self.sample_list = self.sample_list[:int(len(self.sample_list) * args.data_ratio)]
        self.total = len(self.sample_list)
        self.transform = transforms.Compose(
            [
                transforms.RandCropByPosNegLabeld(
                    spatial_size=(96, 96, 96),
                    keys=["image"],
                    label_key="image",
                    pos=6,
                    neg=2,
                    num_samples=8,
                ),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
                transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
                transforms.RandRotate90d(keys=["image"], prob=0.3, max_k=3),
                transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
                transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                transforms.ToTensord(keys=["image"]),
            ]
        )
        print("Total: {} pretrained 3d volumes".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = torch.load(self.sample_list[idx])
        sample = self.transform({"image": case["image"]}) 

        return sample

