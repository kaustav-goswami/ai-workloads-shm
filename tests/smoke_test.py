import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    print("Building model with PyTorch's default (heap) allocator...")
    model = SimpleModel()

    with torch.no_grad():
        model.fc1.weight.fill_(1.0)
        model.fc1.bias.fill_(0.5)
        model.fc2.weight.fill_(2.0)
        model.fc2.bias.fill_(1.0)

    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    print(f"Input tensor x: {x}")

    with torch.no_grad():
        out = model(x)

    print(f"Output tensor: {out}")

    expected = torch.tensor([[37.0, 37.0]])
    if torch.allclose(out, expected):
        print("SUCCESS! Model weights initialized and forward pass verified correctly.")
    else:
        raise RuntimeError(f"FAILURE! Output {out} does not match expected {expected}.")


if __name__ == "__main__":
    main()
