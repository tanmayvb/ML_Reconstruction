import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------
# Dice Loss (with logits support)
# ---------------------------------------
def dice_loss(pred, target, smooth=1e-5):

    pred = torch.sigmoid(pred)  # 🔥 important for logits

    pred = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum()

    return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


# ---------------------------------------
# Combined Dice + BCEWithLogits
# ---------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=0.7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = dice_loss(pred, target)
        #print(f"BCE={bce.item():.4f}, Dice={dice.item():.4f}")

        return self.bce_weight * bce + self.dice_weight * dice

def get_loss(name="dice_bce"):

    if name == "dice_bce":
        return DiceBCELoss()

    elif name == "bce":
        return nn.BCEWithLogitsLoss()

    elif name == "mse":
        return nn.MSELoss()

    elif name == "l1":
        return nn.L1Loss()

    elif name == "cross_entropy":
        return nn.CrossEntropyLoss()

    else:
        raise ValueError(f"Unknown loss: {name}")
