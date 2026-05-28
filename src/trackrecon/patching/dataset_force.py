import torch
from torch.utils.data import Dataset
import numpy as np
import tifffile as tiff

class SpineDatasetForce(Dataset):
    def __init__(self, img_path, mask_path, patch_size=(48,128,128)):
        self.img = tiff.imread(img_path)
        self.mask = tiff.imread(mask_path)
        self.patch_size = patch_size

        self.Z, self.Y, self.X = self.img.shape

    def __len__(self):
        #return 200  # patches per epoch
        return 10

    def __getitem__(self, idx):
        pz, py, px = self.patch_size

        # 50% chance: sample near spine
        if np.random.rand() < 0.7:
            coords = np.argwhere(self.mask > 0.1)

            zc, yc, xc = coords[np.random.randint(len(coords))]

            z = np.clip(zc - pz//2, 0, self.Z - pz)
            y = np.clip(yc - py//2, 0, self.Y - py)
            x = np.clip(xc - px//2, 0, self.X - px)

        else:
            # random patch
            z = np.random.randint(0, self.Z - pz)
            y = np.random.randint(0, self.Y - py)
            x = np.random.randint(0, self.X - px)

        img_patch = self.img[z:z+pz, y:y+py, x:x+px]
        mask_patch = self.mask[z:z+pz, y:y+py, x:x+px]

        # normalize
        img_patch = img_patch.astype(np.float32)
        img_patch = (img_patch - img_patch.min()) / (img_patch.max() - img_patch.min() + 1e-8)

        if np.random.rand() > 0.5:
            img_patch = np.flip(img_patch, axis=1)
            mask_patch = np.flip(mask_patch, axis=1)

        k = np.random.randint(0, 4)
        img_patch = np.rot90(img_patch, k, axes=(1,2))
        mask_patch = np.rot90(mask_patch, k, axes=(1,2))

        return (
            torch.tensor(img_patch).unsqueeze(0),
            torch.tensor(mask_patch).float().unsqueeze(0)
        )
