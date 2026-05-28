import pandas as pd
import numpy as np
from pathlib import Path


def save_points_csv(points, path, scores=None):
    """
    Save points to CSV.

    points: (N,3) array or list of (z,y,x)
    scores: optional (N,) array
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points)

    if points.size == 0:
        df = pd.DataFrame(columns=["z", "y", "x"])
    else:
        df = pd.DataFrame(points, columns=["z", "y", "x"])

    if scores is not None:
        df["score"] = scores

    df.to_csv(path, index=False)

    print(f"[INFO] Saved {len(df)} points → {path}")


def load_points_csv(path, return_scores=False):
    """
    Load points from CSV.

    returns:
        points (N,3)
        optionally scores
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Points file not found: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        points = np.empty((0, 3))
        scores = np.empty((0,))
    else:
        points = df[["z", "y", "x"]].values
        scores = df["score"].values if "score" in df.columns else None

    if return_scores:
        return points, scores
    else:
        return points
