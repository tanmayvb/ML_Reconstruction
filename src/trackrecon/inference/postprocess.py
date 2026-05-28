import numpy as np
from skimage.measure import label, regionprops


def extract_spines(prediction, threshold=0.8, min_size=20):
    print("\nExtracting spines from predictions")

    print(
        "Prediction stats:",
        prediction.min(),
        prediction.max(),
        prediction.mean()
    )
    print("Is NaN= ", np.isnan(prediction).sum())
    mask = prediction > threshold

    print(
        "Foreground voxels:",
        mask.sum()
    )

    print(
        "Foreground fraction:",
        mask.mean()
    )

    labeled = label(mask)

    print("Connected component labeling done")

    spines = []

    for region in regionprops(labeled):
        if region.area < min_size:
            continue

        z, y, x = region.centroid

        spines.append({
            "centroid": (z, y, x),
            "volume": region.area
        })
    
    print("Sucessfully detected spines\n")
    return spines, labeled
