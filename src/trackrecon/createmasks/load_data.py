import tifffile as tiff
import numpy as np

import tifffile as tiff
import numpy as np


def load_volume(
        path,
        t=0,
        c=0
    ):

    img = tiff.imread(path)

    print(f"[DEBUG] Original shape: {img.shape}")

    # ---------------------------------
    # ZYX
    # ---------------------------------
    if img.ndim == 3:

        volume = img

    # ---------------------------------
    # CZYX or ZCYX
    # ---------------------------------
    elif img.ndim == 4:

        # assume CZYX
        if img.shape[0] <= 4:

            volume = img[c]

        # assume ZCYX
        elif img.shape[1] <= 4:

            volume = img[:, c]

        else:
            raise ValueError(
                f"Ambiguous 4D shape: {img.shape}"
            )

    # ---------------------------------
    # TCZYX
    # ---------------------------------
    elif img.ndim == 5:

        volume = img[t, c]

    else:

        raise ValueError(
            f"Unsupported shape: {img.shape}"
        )

    volume = volume.astype(np.float32)

    # normalize
    volume = (
        volume - volume.min()
    ) / (
        volume.max() - volume.min() + 1e-8
    )

    print(f"[DEBUG] Final shape: {volume.shape}")

    print(
        f"[DEBUG] Min={volume.min():.4f} "
        f"Max={volume.max():.4f} "
        f"Mean={volume.mean():.4f}"
    )

    return volume

