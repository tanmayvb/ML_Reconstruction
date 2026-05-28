import numpy as np

def evaluate(pred_points, gt_points, tol=3):

    matched = 0

    for p in pred_points:
        dists = np.linalg.norm(gt_points - p, axis=1)
        if np.min(dists) < tol:
            matched += 1

    precision = matched / len(pred_points) if len(pred_points) else 0
    recall = matched / len(gt_points) if len(gt_points) else 0

    return precision, recall
