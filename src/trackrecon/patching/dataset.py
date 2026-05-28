import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile as tiff

class SpineDataset(Dataset):
    def __init__(self, img_path, mask_path, patch_size=(32,128,128)):
        self.img = tiff.imread(img_path)
        self.mask = tiff.imread(mask_path)
        self.patch_size = patch_size

        self.Z, self.Y, self.X = self.img.shape

    def __len__(self):
        return 200  # patches per epoch

    def __getitem__(self, idx):
        pz, py, px = self.patch_size

        z = np.random.randint(0, self.Z - pz)
        y = np.random.randint(0, self.Y - py)
        x = np.random.randint(0, self.X - px)

        img_patch = self.img[z:z+pz, y:y+py, x:x+px]
        mask_patch = self.mask[z:z+pz, y:y+py, x:x+px]

        img_patch = img_patch.astype(np.float32)
        img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min() + 1e-8)

        return (
            torch.tensor(img_patch).unsqueeze(0),
            torch.tensor(mask_patch).float().unsqueeze(0)
        )
