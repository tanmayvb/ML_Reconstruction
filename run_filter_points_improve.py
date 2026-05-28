import pandas as pd
import tifffile as tiff
from filter_points import filter_points_advanced

# load
volume = tiff.imread("volume.tiff")
df = pd.read_csv("points.csv")

points = df.iloc[:, -3:].values

# run filtering
filtered_points, scores = filter_points_advanced(
    volume,
    points,
    percentile=70,
    min_distance=3
)

print("Original:", len(points))
print("Filtered:", len(filtered_points))

# save
pd.DataFrame(filtered_points, columns=["axis-0","axis-1","axis-2"]) \
    .to_csv("filtered_points_advanced.csv", index=False)
