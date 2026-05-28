import torch
import torch.nn as nn

def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, 3, padding=1),
        nn.ReLU(),
        nn.Conv3d(out_c, out_c, 3, padding=1),
        nn.ReLU(),
    )

class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = conv_block(128, 256)

        self.up3 = nn.ConvTranspose3d(256, 128, 2, 2)
        self.dec3 = conv_block(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, 2, 2)
        self.dec2 = conv_block(128, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, 2, 2)
        self.dec1 = conv_block(64, 32)

        self.out = nn.Conv3d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        #return torch.sigmoid(self.out(d1))
        return self.out(d1)
