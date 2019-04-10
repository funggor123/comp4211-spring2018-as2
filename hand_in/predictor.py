import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, hidden_num=20):
        super(Predictor, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(32, hidden_num),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(hidden_num, 47)

    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


