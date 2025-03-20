import torch
from torch import nn


class CovidCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 3, 1, 1),

            nn.MaxPool2d(2, 2),
            nn.Linear(128 * 28 * 28, 256),
        nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.model(x)
        return x

if __name__ == '__main__':
    model = CovidCNN(num_classes=3)
print(model)
