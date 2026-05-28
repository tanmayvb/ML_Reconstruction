from torch.utils.data import DataLoader

from ..patching.dataset_force import SpineDatasetForce
from ..patching.dataset_multivolume import MultiVolumeDataset
from ..models.model_3dunet import UNet3D
from ..models.loss_fun import dice_loss, loss_fn
import torch
import matplotlib.pyplot as plt


dataset = SpineDatasetForce(
    "/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/processed/EH123_20260908_Airyscan-mode_Resolusion-weightened.ome.tiff",
    "/Users/tanmay/LenevoINFN/Work/UTokyo/V1/MLtorch/ML_Reconstruction/src/trackrecon/data/levels/EH123_20260908_Airyscan-mode_Resolusion-weightened_mask.tiff"
)

#-------------------------------
# MultiVolume
#-------------------------------
image_paths = [
    "data/processed/vol1.tiff",
    "data/processed/vol2.tiff",
    "data/processed/vol3.tiff"
]

mask_paths = [
    "data/labels/vol1_mask.tiff",
    "data/labels/vol2_mask.tiff",
    "data/labels/vol3_mask.tiff"
]

#----------------------------------
#Autoloader
#----------------------------------
image_paths = sorted(glob.glob("data/processed/*.tiff"))
mask_paths = sorted(glob.glob("data/labels/*_mask.tiff"))
#-----------------------------------------------------------

dataset = MultiVolumeDataset(image_paths, mask_paths)
#-----------------------------------------------------------

loader = DataLoader(dataset, batch_size=1, shuffle=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

model = UNet3D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

loss_history = []

for epoch in range(50):
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        #loss = dice_loss(pred, y)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch}: Loss = {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "model.pth")

plt.figure(figsize=(6,4))
plt.plot(loss_history, '-o', linewidth=2)
#plt.ylim(0, 1)
plt.title("Initial Training Behaviour (Prototype)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("loss_curve_optimise.png")
plt.show()
