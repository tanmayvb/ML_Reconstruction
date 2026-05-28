import numpy as np
from scipy.spatial import distance


def match_spines(
        pred_spines,
        filtered_points,
        max_dist=5
    ):

    # ---------------------------------
    # Convert to arrays
    # ---------------------------------
    pred_points = np.array(
        [s["centroid"] for s in pred_spines]
    )

    filtered_points = np.array(
        filtered_points
    )

    # ---------------------------------
    # Safety checks
    # ---------------------------------
    if len(pred_points) == 0:

        print("[WARN] No predicted spines")

        return 0.0, 0.0

    if len(filtered_points) == 0:

        print("[WARN] No filtered points")

        return 0.0, 0.0

    print(f"Predicted points: {len(pred_points)}")
    print(f"Filtered points : {len(filtered_points)}")

    matched = 0

    used = set()

    # ---------------------------------
    # Match predictions
    # ---------------------------------
    for p in pred_points:

        dists = distance.cdist(
            [p],
            filtered_points
        )[0]

        idx = np.argmin(dists)

        if (
            dists[idx] < max_dist
            and idx not in used
        ):

            matched += 1

            used.add(idx)

    # ---------------------------------
    # Metrics
    # ---------------------------------
    precision = matched / len(pred_points)

    recall = matched / len(filtered_points)

    print(f"[DEBUG] Matched Points: {matched}")

    return precision, recall
