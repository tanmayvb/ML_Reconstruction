from skimage.feature import peak_local_max

def auto_points(volume, min_distance, threshold_abs):
    coords = peak_local_max(
        volume,
        min_distance=3,
        threshold_abs=0.3
    )
    return coords
