import numpy as np
import torch


#=================================
# threshold tuning (big impact)
# min_size tuning
# sigma (Gaussian mask)
# stride vs patch size
#=================================

def predict_full_volume(model, volume,
                        patch_size=(32,64,64),
                        stride=(32,64,64),
                        device="cpu"):

    print("Start prediction")
    model.eval()
    print(next(model.parameters()).mean())
    print("Param mean:", next(model.parameters()).mean())
    print("NaNs in volume:", np.isnan(volume).any())

    Z, Y, X = volume.shape
    pz, py, px = patch_size
    sz, sy, sx = stride

    assert Z >= pz and Y >= py and X >= px, "Volume smaller than patch size"

    output = np.zeros_like(volume, dtype=np.float32)
    count = np.zeros_like(volume, dtype=np.float32)

    z_steps = sorted(set(list(range(0, Z - pz + 1, sz)) + [Z - pz]))
    y_steps = sorted(set(list(range(0, Y - py + 1, sy)) + [Y - py]))
    x_steps = sorted(set(list(range(0, X - px + 1, sx)) + [X - px]))

    total = len(z_steps) * len(y_steps) * len(x_steps)
    counter = 0

    with torch.no_grad():
        for z in z_steps:
            for y in y_steps:
                for x in x_steps:

                    counter += 1
                    if counter % 10 == 0:
                        print(f"[DEBUG] Patch {counter}/{total}")

                    patch = volume[z:z+pz, y:y+py, x:x+px]

                    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)

                    pred = torch.sigmoid(model(patch)).cpu().numpy()[0, 0]
                    #print(f"[DEBUG] Prediction torch from prediction full volume: {pred.min()}, {pred.max()}")

                    output[z:z+pz, y:y+py, x:x+px] += pred
                    count[z:z+pz, y:y+py, x:x+px] += 1

    return output / (count + 1e-8)

