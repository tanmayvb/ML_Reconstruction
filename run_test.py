from torch.utils.data import DataLoader
from src.trackrecon.data.dataset import SpinePatchDataset



dataset = SpinePatchDataset("/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/Airyscan_data/EH123_20260908_Airyscan-mode_Resolusion-weightened.czi")

print("Total patches:", len(dataset))

loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in loader:
    print("Batch shape:", batch.shape)
    break
