import matplotlib.pyplot as plt
import tifffile as tiff
import pandas as pd
from src.trackrecon.data.load_data import load_volume
from src.trackrecon.createmasks.create_mask import create_mask

volume = load_volume("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/processed/EH123_20260908_Airyscan-mode_Resolusion-weightened.ome.tiff")
z = volume.shape[0] // 2

df = pd.read_csv("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/createmasks/filtered_points.csv").values
print(df)  # debug
points = df[:, -3:]
mask = create_mask(volume.shape, points)

plt.figure(figsize=(6,6))
plt.imshow(volume[z], cmap="viridis")
plt.imshow(mask[z], cmap="Greens", alpha=0.8)
plt.title("Overlay (Image + Mask)")
plt.colorbar()
plt.savefig("Ovelat_plot_raw_filterpoints.png")
plt.show()
