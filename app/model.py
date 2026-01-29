import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

class IntrinsicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ConvBlock(3, 64)
        self.normals = nn.Conv2d(64, 3, 1)
        self.diffuse = nn.Conv2d(64, 3, 1)
        self.specular = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        f = self.encoder(x)
        normals = F.normalize(self.normals(f), dim=1)
        diffuse = torch.sigmoid(self.diffuse(f))
        specular = torch.sigmoid(self.specular(f))
        return normals, diffuse, specular
