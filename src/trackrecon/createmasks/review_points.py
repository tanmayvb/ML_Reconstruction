import napari
import pandas as pd
from trackrecon.datasets.load_data import load_volume

volume = load_volume("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/Airyscan_data/EH123_20260908_Airyscan-mode_Resolusion-weightened.tif")
points = pd.read_csv("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/auto_points.csv").values

viewer = napari.Viewer()
viewer.add_image(volume)
viewer.add_points(points, size=5, face_color="red")

napari.run()
