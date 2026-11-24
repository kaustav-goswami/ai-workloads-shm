import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.common.seeds import deterministic_seed
from benchmark.common.timing import TimingRecorder


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
    """A small conv-only image classifier. Its weights are a handful of
    large, dense 4D conv tensors (out_ch, in_ch, kh, kw) rather than the many
    small 2D Linear tensors the other workloads use - a genuinely different
    allocation shape.
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


def build_model(config=None):
    config = default_config() if config is None else config
    model = SimpleCNN(
        in_channels=config["in_channels"],
        stem_channels=config["stem_channels"],
        num_blocks=config["num_blocks"],
        num_classes=config["num_classes"],
    )
    return model.to(config["dtype"])


def initialize_weights(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(deterministic_seed(name))
            if name.endswith(".bias"):
                param.zero_()
            else:
                fan_in = param.shape[1:].numel()
                bound = (1.0 / fan_in) ** 0.5
                param.copy_(
                    (torch.rand(param.shape, generator=generator, dtype=param.dtype) * 2 - 1) * bound
                )


def local_image_batch(config=None):
    config = default_config() if config is None else config
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    return torch.randn(
        config["batch_size"],
        config["in_channels"],
        config["image_size"],
        config["image_size"],
        generator=generator,
        dtype=config["dtype"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="PT-Vision: conv-only CNN. Weights (a handful of large 4D conv "
        "tensors) are ordinary tensors allocated with PyTorch's default CPU allocator."
    )
    parser.parse_args()

    print("PT-Vision: building conv model and initializing weights...")
    config = default_config()
    recorder = TimingRecorder(label="PT-Vision")
    with recorder.time("build"):
        model = build_model(config)
    with recorder.time("init"):
        initialize_weights(model)

    x = local_image_batch(config)
    print(f"PT-Vision: running forward pass on image batch {tuple(x.shape)}...")
    with recorder.time("forward"):
        with torch.no_grad():
            out = model(x)
    recorder.print_summary()

    print(f"PT-Vision: output shape {tuple(out.shape)}, mean {out.float().mean().item():.6f}")
    print("PT-Vision: SUCCESS - conv forward pass completed.")


if __name__ == "__main__":
    main()
