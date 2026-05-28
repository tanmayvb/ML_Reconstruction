import tifffile as tiff
import numpy as np

def load_volume(path):
    img = tiff.imread(path)

    # handle (C, Z, Y, X)
    if img.ndim == 4:
        img = img[0]   # choose channel

    img = img.astype("float32")
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    print("Final shape:", img.shape)  # should be (Z,Y,X)

    return img
