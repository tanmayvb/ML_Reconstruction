import numpy as np
import pandas as pd
import tifffile as tiff


def filter_points(volume, points, threshold_ratio=1.5):
    """
    volume: (Z,Y,X)
    points: (N,3)
    threshold_ratio: how much brighter than background
    """

    Z, Y, X = volume.shape
    global_mean = volume.mean()

    good_points = []
    scores = []

    for (z, y, x) in points:
        z, y, x = int(z), int(y), int(x)

        # boundary check
        if z < 1 or y < 1 or x < 1 or z >= Z-1 or y >= Y-1 or x >= X-1:
            continue

        patch = volume[z-1:z+2, y-1:y+2, x-1:x+2]

        local_mean = patch.mean()
        score = local_mean / (global_mean + 1e-8)

        scores.append(score)

        if score > threshold_ratio:
            good_points.append([z, y, x])

    return np.array(good_points), np.array(scores)



def filter_points_percentile(volume, points, percentile=60):

    Z, Y, X = volume.shape
    global_mean = volume.mean()

    scores = []
    valid_points = []

    for (z, y, x) in points:
        z, y, x = int(z), int(y), int(x)

        if z < 1 or y < 1 or x < 1 or z >= Z-1 or y >= Y-1 or x >= X-1:
            continue

        patch = volume[z-1:z+2, y-1:y+2, x-1:x+2]
        local_mean = patch.mean()
        score = local_mean / (global_mean + 1e-8)

        scores.append(score)
        valid_points.append([z, y, x])

    scores = np.array(scores)
    valid_points = np.array(valid_points)

    threshold = np.percentile(scores, percentile)

    filtered = valid_points[scores > threshold]

    return filtered, scores, threshold
