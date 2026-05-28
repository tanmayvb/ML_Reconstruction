import numpy as np
import tifffile as tiff
import pandas as pd

#----------------------------------------------
#Run:
#uv run python run_create_gaussian_mask.py
#----------------------------------------------

#----------------------------------------------
#Recommended parameters
#sigma_z  = 2
#sigma_xy = 4–6  important!!
# Dependence on input file
#----------------------------------------------

def gaussian_mask(volume_shape, points, sigma_z=1.5, sigma_xy=5):

    Z, Y, X = volume_shape
    mask = np.zeros(
               (Z, Y, X), 
               dtype=np.float32
           )
    print("Gaussian mask shape:", mask.shape)
    # precompute small kernel size
    kz = int(3 * sigma_z)
    kxy = int(3 * sigma_xy)

    for (z, y, x) in points:
        z, y, x = int(z), int(y), int(x)

        # local region
        z_min, z_max = max(0, z - kz), min(Z, z + kz + 1)
        y_min, y_max = max(0, y - kxy), min(Y, y + kxy + 1)
        x_min, x_max = max(0, x - kxy), min(X, x + kxy + 1)

        # coordinate grids
        zz, yy, xx = np.meshgrid(
            np.arange(z_min, z_max) - z,
            np.arange(y_min, y_max) - y,
            np.arange(x_min, x_max) - x,
            indexing='ij'
        )

        # Gaussian
        g = np.exp(
                -(zz**2/(2*sigma_z**2)
                + yy**2/(2*sigma_xy**2)
                + xx**2/(2*sigma_xy**2))
        )
        
        g[g < 1e-4] = 0  # remove tiny values
   
        mask[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(
            mask[z_min:z_max, y_min:y_max, x_min:x_max],
            g
        )

    # normalize
    mask = mask / (mask.max() + 1e-8)

    return mask
