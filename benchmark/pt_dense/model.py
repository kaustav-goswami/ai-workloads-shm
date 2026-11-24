import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.common.seeds import deterministic_seed
from benchmark.common.timing import TimingRecorder


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + 1e-5)
        return self.weight * x


class LlamaAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = LlamaAttention(hidden_size, num_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def forward(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


class LlamaDenseModel(nn.Module):
    def __init__(self, hidden_size=512, num_layers=4, num_heads=8, intermediate_size=1376):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(hidden_size, num_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def default_config():
    return {
        "hidden_size": 512,
        "num_layers": 4,
        "num_heads": 8,
        "intermediate_size": 1376,
        "dtype": torch.float16,
        "batch_size": 2,
        "seq_len": 16,
    }


def build_model(config=None):
    config = default_config() if config is None else config
    model = LlamaDenseModel(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        intermediate_size=config["intermediate_size"],
    )
    return model.to(config["dtype"])


def initialize_weights(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(deterministic_seed(name))
            if "norm" in name or name.endswith(".weight") and param.ndim == 1:
                param.fill_(1.0)
            else:
                param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.02)


def local_activation_input(config=None):
    config = default_config() if config is None else config
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    return torch.randn(
        config["batch_size"],
        config["seq_len"],
        config["hidden_size"],
        generator=generator,
        dtype=config["dtype"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="PT-Dense: monolithic LLaMA-style dense transformer. Weights and "
        "activations are ordinary tensors allocated with PyTorch's default CPU allocator."
    )
    parser.parse_args()

    print("PT-Dense: building model and initializing weights...")
    config = default_config()
    recorder = TimingRecorder(label="PT-Dense")
    with recorder.time("build"):
        model = build_model(config)
    with recorder.time("init"):
        initialize_weights(model)

    x = local_activation_input(config)
    print(f"PT-Dense: running forward pass on activations {tuple(x.shape)}...")
    with recorder.time("forward"):
        with torch.no_grad():
            out = model(x)
    recorder.print_summary()

    print(f"PT-Dense: output shape {tuple(out.shape)}, mean {out.float().mean().item():.6f}")
    print("PT-Dense: SUCCESS - dense GEMM forward pass completed.")


if __name__ == "__main__":
    main()
