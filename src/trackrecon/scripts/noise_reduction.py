from scipy.ndimage import gaussian_filter

def denoise_volume(
        volume,
        sigma=(1,1,1)
    ):

    return gaussian_filter(
        volume,
        sigma=sigma
    )
