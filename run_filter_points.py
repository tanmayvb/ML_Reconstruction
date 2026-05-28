import pandas as pd
import numpy as np
import tifffile as tiff
from filter_points import filter_points, filter_points_percentile
import matplotlib.pyplot as plt

# load
inFile = "/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/processed/EH123_20260908_Airyscan-mode_Resolusion-weightened.ome.tiff"
volume = tiff.imread(inFile)
points = pd.read_csv("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/createmasks/auto_points_EH123_20260908_Airyscan-mode_Resolusion-weightened.csv").values[:, -3:]

# normalize (important!)
volume = volume.astype(np.float32)
volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

"""
# filter
filtered_points, scores = filter_points(volume, points, threshold_ratio=1.2)

print("Original points:", len(points))
print("Filtered points:", len(filtered_points))
print("Removed:", len(points) - len(filtered_points))
"""

filtered_points, scores, thr = filter_points_percentile(volume, points, 60)

print("Threshold:", thr)
print("Original:", len(points))
print("Filtered:", len(filtered_points))



# save
pd.DataFrame(filtered_points, columns=["axis-0","axis-1","axis-2"]) \
    .to_csv("filtered_points.csv", index=False)

plt.hist(scores, bins=60)
plt.title("Point intensity scores")
plt.xlabel("score")
plt.ylabel("count")
plt.show()
