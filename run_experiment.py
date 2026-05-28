import pandas as pd
import tifffile as tiff
from src.trackrecon.data.load_data import load_volume
from src.trackrecon.createmasks.auto_detect import detect_spines
from src.trackrecon.createmasks.create_mask import create_mask


inFile = "/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/processed/EH123_20260908_Airyscan-mode_Resolusion-weightened.ome.tiff"
volume = load_volume(inFile)

points = detect_spines(volume)
print("Detected:", len(points))

pd.DataFrame(points, columns=["axis-0","axis-1","axis-2"]).to_csv(
    "/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/createmasks/auto_points_EH123_20260908_Airyscan-mode_Resolusion-weightened.csv", index=False
)

"""
df = pd.read_csv("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/cleaned_points.csv").values
print(df)  # debug
points = df[:, -3:]

print(points[:5])
print(points.shape)


mask = create_mask(volume.shape, points)

tiff.imwrite("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/processedmask.tiff", mask.astype("float32"))
"""
