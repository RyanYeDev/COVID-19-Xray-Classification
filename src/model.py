import torch
from torch import nn


class CovidCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 112 * 112, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.model(x)
        return x

# if __name__ == '__main__':
#     model = CovidCNN(num_classes=3)
#     print(model)
#
#     sample_input = torch.randn(1, 1, 224, 224)
#     output = model(sample_input)
#     print("Output shape:", output.shape)
#     print(model(sample_input).shape)
