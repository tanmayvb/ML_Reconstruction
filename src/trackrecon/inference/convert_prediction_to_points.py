from skimage.feature import peak_local_max

def detect_spines(pred, threshold=0.5, min_distance=3):

    coords = peak_local_max(
        pred,
        threshold_abs=threshold,
        min_distance=min_distance
    )

    return coords
