import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(8, 4, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out
