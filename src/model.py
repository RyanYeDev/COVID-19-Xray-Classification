import torch
from torch import nn

class Covid_cnn(nn.Module):
    def __init__(self, num_classes=2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model1 = nn.Sequential(
            # block 1
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # block 2
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            # block 3
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            # block 4
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # block 5
            nn.Conv2d(128, 256, 3, 1, 1),  # 普通卷积
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  # 普通卷积
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            # Flatten 和 Linear
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, num_classes)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

if __name__ == '__main__':
    covid_cnn = Covid_cnn().cuda()
    input = torch.ones(1, 3, 224, 224).cuda()
    print(input.shape)
    output = covid_cnn(input)
    print(output.shape)