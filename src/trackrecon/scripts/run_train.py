import torch
from torch.utils.data import DataLoader

from patching.dataset import SpineDataset
from models.unet3d import UNet3D
from models.losses import dice_loss


def train():

    dataset = SpineDataset(
        image_path="data/raw/image.ome.tiff",
        label_path="data/labels/mask.ome.tiff",
        patch_size=(64,64,64),
        n_samples=2000
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = dice_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: Loss = {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()
