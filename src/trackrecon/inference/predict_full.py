import numpy as np
import torch
import tifffile as tiff
from trackrecon.models.unet3d import UNet3D
from src.trackrecon.data.load_data import load_volume

"""
def load_volume(path, t=0, c=0):
    img = tiff.imread(path)

    if img.ndim == 5:
        img = img[t, :, c, :, :]
    elif img.ndim == 4:
        img = img[:, c, :, :]
    else:
        raise ValueError("Unsupported shape")

    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return img
"""

def predict_full(model, volume, patch_size=(48,128,128), stride=(24,64,64), device="cpu"):
#def predict_full_volume(model, volume, patch_size=(64,64,64), stride=(32,32,32), device="cuda"):
    model.eval()

    Z, Y, X = volume.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    output = np.zeros((Z, Y, X), dtype=np.float32)
    count = np.zeros((Z, Y, X), dtype=np.float32)

    with torch.no_grad():
        for z in range(0, Z - pz + 1, sz):
            for y in range(0, Y - py + 1, sy):
                for x in range(0, X - px + 1, sx):

                    patch = volume[z:z+pz, y:y+py, x:x+px]
                    patch = torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(device)

                    pred = model(patch).cpu().numpy()[0, 0]

                    output[z:z+pz, y:y+py, x:x+px] += pred
                    count[z:z+pz, y:y+py, x:x+px] += 1

    return output / (count + 1e-8)


def run_inference(model_path, image_path):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = UNet3D().to(device)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    volume = load_volume(image_path)

    prediction = predict_full_volume(model, volume, device=device)

    return volume, prediction
