# scripts/evaluate_81_combinations.py

import tifffile as tiff
import pandas as pd

from itertools import product

thresholds = [
    0.7,
    0.8,
    0.9
]

min_sizes = [
    20,
    50,
    100
]

models = [

    "train_lr_3e-05_patches_100",
    "train_lr_3e-05_patches_200",
    "train_lr_3e-05_patches_400",

    "train_lr_1e-04_patches_100",
    "train_lr_1e-04_patches_200",
    "train_lr_1e-04_patches_400",

    "train_lr_3e-04_patches_100",
    "train_lr_3e-04_patches_200",
    "train_lr_3e-04_patches_400"
]

results = []

for model_dir in models:

    prediction = tiff.imread(
        f"{model_dir}/prediction_volume.ome.tiff"
    )

    for threshold, min_size in product(
            thresholds,
            min_sizes
        ):

        pred_spines, labeled = extract_spines(
            prediction,
            threshold=threshold,
            min_size=min_size
        )

        precision, recall = match_spines(
            pred_spines,
            filtered_points
        )

        f1 = (
            2 * precision * recall /
            (precision + recall + 1e-8)
        )

        results.append({

            "model": model_dir,

            "threshold": threshold,

            "min_size": min_size,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "detections": len(pred_spines)
        })

df = pd.DataFrame(results)

df.to_csv(
    "grid_scan_results.csv",
    index=False
)

print(
    "\n===== TOP RESULTS =====\n"
)

print(
    df.sort_values(
        "f1",
        ascending=False
    ).head(20)
)
