import sys
import napari
from src.trackrecon.data.dataset import SpinePatchDataset


print(sys.executable)
dataset = SpinePatchDataset("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/Airyscan_data/EH123_20260908_Airyscan-mode_Resolusion-weightened.czi")

patch = dataset[0].numpy()[0]  # (Z, Y, X)

viewer = napari.Viewer()
viewer.add_image(patch, name="patch")

napari.run()
