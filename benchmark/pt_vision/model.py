import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = self.conv2(h)
        return F.relu(x + h)


class SimpleCNN(nn.Module):
    """A small conv-only image classifier. Its shared weights are a handful
    of large, dense 4D conv tensors (out_ch, in_ch, kh, kw) rather than the
    many small 2D Linear tensors the other workloads share - a genuinely
    different allocation shape to fragment/allocate against.
    """

    def __init__(self, in_channels=3, stem_channels=64, num_blocks=4, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, stem_channels, kernel_size=3, padding=1, bias=True)
        self.blocks = nn.ModuleList([ConvBlock(stem_channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stem_channels, num_classes)

    def forward(self, x):
        x = F.relu(self.stem(x))
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def default_config():
    return {
        "in_channels": 3,
        "stem_channels": 64,
        "num_blocks": 4,
        "num_classes": 10,
        "dtype": torch.float32,
        "batch_size": 4,
        "image_size": 32,
    }
