import torch
from torch.utils.data import Dataset
import numpy as np

class SpineDataset(Dataset):
    def __init__(self, volume, mask, patch_size=(32,128,128), n_samples=2000):
        self.volume = volume
        self.mask = mask
        self.patch_size = patch_size
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        Z,Y,X = self.volume.shape
        pz,py,px = self.patch_size

        z = np.random.randint(0, Z-pz)
        y = np.random.randint(0, Y-py)
        x = np.random.randint(0, X-px)

        img = self.volume[z:z+pz, y:y+py, x:x+px]
        lbl = self.mask[z:z+pz, y:y+py, x:x+px]

        return (
            torch.tensor(img).unsqueeze(0),
            torch.tensor(lbl).unsqueeze(0)
        )
