import numpy as np


def crop_volume(
        volume,
        z0, z1,
        y0, y1,
        x0, x1
    ):

    cropped = volume[
        z0:z1,
        y0:y1,
        x0:x1
    ]

    print(
        f"[DEBUG] Cropped shape: {cropped.shape}"
    )

    return cropped
