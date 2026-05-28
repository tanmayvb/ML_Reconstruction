from scipy.ndimage import gaussian_filter

def suppress_background(
        volume,
        sigma=(10,30,30)
    ):

    background = gaussian_filter(
        volume,
        sigma=sigma
    )

    volume = volume - background

    volume = np.clip(volume, 0, None)

    return volume
