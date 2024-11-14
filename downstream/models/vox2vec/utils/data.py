import random

from torch.utils.data import Dataset


class VanillaDataset(Dataset):
    def __init__(self, ids, load_func):
        self.ids = ids
        self.load_func = load_func

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return self.load_func(self.ids[i])


class ResizeByRandomSampling(Dataset):
    def __init__(self, dataset: Dataset, size: int):
        self.dataset = dataset
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        i = random.randint(0, len(self.dataset) - 1)
        return self.dataset[i]
