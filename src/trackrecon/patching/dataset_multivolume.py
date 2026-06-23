import torch
from torch.utils.data import Dataset

import numpy as np
import tifffile as tiff


class MultiVolumeDataset(Dataset):

    def __init__(
            self,
            image_paths,
            mask_paths,
            positive_sampling_ratio,
            patches_per_epoch,
            patch_size=(48, 128, 128),
    ):

        print("[DEBUG] Initializing MultiVolumeDataset")

        self.positive_sampling_ratio = positive_sampling_ratio
        self.patches_per_epoch = patches_per_epoch
        self.patch_size = patch_size

        # ---------------------------------
        # Load images
        # ---------------------------------
        self.images = []

        for p in image_paths:

            img = tiff.imread(p)

            img = img.astype(np.float32)

            img = (
                img - img.min()
            ) / (
                img.max() - img.min() + 1e-8
            )

            self.images.append(img)

            print(f"[DEBUG] Loaded image : {p}")
            print(f"[DEBUG] Image shape  : {img.shape}")

        # ---------------------------------
        # Load masks
        # ---------------------------------
        self.masks = []

        for p in mask_paths:

            mask = tiff.imread(p)

            mask = mask.astype(np.float32)

            mask = mask / (mask.max() + 1e-8)

            self.masks.append(mask)

            print(f"[DEBUG] Loaded mask  : {p}")
            print(f"[DEBUG] Mask shape   : {mask.shape}, {mask.min()}, {mask.max()}, {mask.mean()}")

        # ---------------------------------
        # Sanity checks
        # ---------------------------------
        assert len(self.images) == len(self.masks), \
            "Number of images and masks differ"

        for img, mask in zip(self.images, self.masks):

            assert img.shape == mask.shape, \
                f"Shape mismatch: {img.shape} vs {mask.shape}"

            Z, Y, X = img.shape

            pz, py, px = self.patch_size

            assert Z >= pz, \
                f"Z too small: {Z} < {pz}"

            assert Y >= py, \
                f"Y too small: {Y} < {py}"

            assert X >= px, \
                f"X too small: {X} < {px}"

        # ---------------------------------
        # Precompute positive coordinates
        # ---------------------------------
        self.coords = []

        for mask in self.masks:

            coords = np.argwhere(mask > 0.05)

            self.coords.append(coords)

            print(f"[DEBUG] Positive coords: {len(coords)}")

        print("[DEBUG] Dataset loaded successfully")

    def __len__(self):

        return self.patches_per_epoch * len(self.images)

    def __getitem__(self, idx):

        # ---------------------------------
        # Random volume
        # ---------------------------------
        vid = np.random.randint(len(self.images))

        img = self.images[vid]
        mask = self.masks[vid]

        coords = self.coords[vid]

        Z, Y, X = img.shape

        pz, py, px = self.patch_size

        # ---------------------------------
        # Positive patch sampling
        # ---------------------------------
        use_positive = (
            len(coords) > 0 and
            np.random.rand() < self.positive_sampling_ratio
        )

        if use_positive:

            zc, yc, xc = coords[
                np.random.randint(len(coords))
            ]

            # Z
            if Z <= pz:
                z = 0
            else:
                z = int(np.clip(
                    zc - pz // 2,
                    0,
                    Z - pz
                ))

            # Y
            if Y <= py:
                y = 0
            else:
                y = int(np.clip(
                    yc - py // 2,
                    0,
                    Y - py
                ))

            # X
            if X <= px:
                x = 0
            else:
                x = int(np.clip(
                    xc - px // 2,
                    0,
                    X - px
                ))

        # ---------------------------------
        # Random background patch
        # ---------------------------------
        else:

            # Z
            if Z <= pz:
                z = 0
            else:
                z = np.random.randint(
                    0,
                    Z - pz + 1
                )

            # Y
            if Y <= py:
                y = 0
            else:
                y = np.random.randint(
                    0,
                    Y - py + 1
                )

            # X
            if X <= px:
                x = 0
            else:
                x = np.random.randint(
                    0,
                    X - px + 1
                )

        # ---------------------------------
        # Extract patches
        # ---------------------------------
        img_patch = img[
            z:z+pz,
            y:y+py,
            x:x+px
        ]

        mask_patch = mask[
            z:z+pz,
            y:y+py,
            x:x+px
        ]

        # ---------------------------------
        # Final safety checks
        # ---------------------------------
        assert img_patch.shape == (pz, py, px), \
            f"Bad image patch: {img_patch.shape}"

        assert mask_patch.shape == (pz, py, px), \
            f"Bad mask patch: {mask_patch.shape}"

        # ---------------------------------
        # Augmentations
        # ---------------------------------
        if np.random.rand() > 0.5:

            img_patch = np.flip(
                img_patch,
                axis=1
            ).copy()

            mask_patch = np.flip(
                mask_patch,
                axis=1
            ).copy()

        k = np.random.randint(0, 4)

        img_patch = np.rot90(
            img_patch,
            k,
            axes=(1, 2)
        ).copy()

        mask_patch = np.rot90(
            mask_patch,
            k,
            axes=(1, 2)
        ).copy()

        # ---------------------------------
        # Torch tensors
        # ---------------------------------
        img_patch = torch.from_numpy(
            img_patch
        ).float().unsqueeze(0)

        mask_patch = torch.from_numpy(
            mask_patch
        ).float().unsqueeze(0)

        return img_patch, mask_patch
