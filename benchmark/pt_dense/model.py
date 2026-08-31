import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
