import torch  # type: ignore
import torch.nn as nn  # type: ignore
from copy import deepcopy


class MLP(nn.Module):
    """Flexible MLP for flattened CIFAR-10 images."""

    def __init__(self, hidden_sizes=[1024, 512], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = 3072
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


class CNN(nn.Module):
    """Convolutional network with optional pooling layers."""

    def __init__(self, use_pooling=True):
        super().__init__()
        self.use_pooling = use_pooling
        pool = nn.MaxPool2d(2, stride=2) if use_pooling else nn.Identity()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            deepcopy(pool),
            nn.Dropout2d(0.2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            deepcopy(pool),
            nn.Dropout2d(0.3),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            deepcopy(pool),
            nn.Dropout2d(0.4),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x, return_features=False):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        out = self.adaptive_pool(f3)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        if return_features:
            return logits, (f1, f2, f3)
        return logits
