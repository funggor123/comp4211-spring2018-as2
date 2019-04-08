import torch
import torch.nn as nn

''''''


class Encoder(nn.Module):
    def __init__(self, hidden_num=32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.pool1(out)
        out = self.conv4(out)
        return out
