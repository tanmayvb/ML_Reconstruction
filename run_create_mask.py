import pandas as pd
import tifffile as tiff
from src.trackrecon.data.load_data import load_volume
from src.trackrecon.createmasks.create_mask import create_mask

inFile = "/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/processed/EH123_20260908_Airyscan-mode_Resolusion-weightened.ome.tiff"
volume = load_volume(inFile)

points = pd.read_csv("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/filtered_points.csv").values

mask = create_mask(volume.shape, points)

tiff.imwrite("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/levels/EH123_20260908_Airyscan-mode_Resolusion-weightened_mask.tiff", mask.astype("float32"))
