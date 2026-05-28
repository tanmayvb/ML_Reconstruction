import numpy as np

"""
Logical filtering points and need to pass some steps, we need as clean as clean points

raw points : local checks (max + contrast + Z) : compute score :

percentile filtering : distance pruning : final filter points

Need to tune 'contrast_thresh'
"""


def filter_points(volume, points, percentile=70, min_distance=3, contrast_thresh=0.05):

    Z, Y, X = volume.shape
    volume = volume.astype(np.float32)

    global_mean = volume.mean()

    valid_points = []
    scores = []

    # -----------------------------
    # local checks + scoring
    # -----------------------------
    for (z, y, x) in points:
        z, y, x = int(z), int(y), int(x)

        # skip borders use 7×7×7 patches
        if z < 3 or y < 3 or x < 3 or z >= Z-3 or y >= Y-3 or x >= X-3:
            continue

        """
        if z < 1 or y < 1 or x < 1 or z >= Z-1 or y >= Y-1 or x >= X-1:
            continue

        patch = volume[z-1:z+2, y-1:y+2, x-1:x+2]

        local_mean = patch.mean()
        center_val = volume[z, y, x]

        score = local_mean / (global_mean + 1e-8)

        # local maximum condition
        if center_val >= patch.max():
            valid_points.append([z, y, x])
            scores.append(score)
        """

        # -----------------------------
        # local checks + scoring
        # ----------------------------
        patch_small = volume[z-1:z+2, y-1:y+2, x-1:x+2]
        patch_large = volume[z-3:z+4, y-3:y+4, x-3:x+4]

        local_mean = patch_small.mean()
        background = patch_large.mean()
        contrast = local_mean - background

        center_val = volume[z, y, x]

        z_profile = volume[z-2:z+3, y, x]

        if (
            center_val >= patch_small.max()        
            and center_val == z_profile.max()      
            and contrast > contrast_thresh
         ):
             score = contrast
             scores.append(score)     
             valid_points.append([z, y, x])

    valid_points = np.array(valid_points)
    scores = np.array(scores)

    if len(valid_points) == 0:
        return np.empty((0, 3)), scores

    # -----------------------------
    # percentile filtering
    # -----------------------------
    threshold = np.percentile(scores, percentile)

    mask = scores >= threshold
    filtered = valid_points[mask]
    filtered_scores = scores[mask]

    # -----------------------------
    # distance-based pruning
    # -----------------------------

    if len(filtered) == 0:
        return np.empty((0, 3)), filtered_scores

    final_points = []

    used = np.zeros(len(filtered), dtype=bool)

    for i in np.argsort(-filtered_scores):  # highest score first
        if used[i]:
            continue

        p = filtered[i]
        final_points.append(p)

        # suppress nearby points
        dists = np.linalg.norm(filtered - p, axis=1)
        used[dists < min_distance] = True

    final_points = np.array(final_points)

    return final_points, filtered_scores
