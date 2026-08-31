import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.pt_dense.model import RMSNorm


class QuantLinear(nn.Module):
    """A Linear layer whose weight lives as int8 + a per-output-channel fp32
    scale (symmetric quantization, no zero-point), dequantized on the fly in
    forward(). qweight/scale are buffers (not Parameters) so they show up in
    state_dict() and can be shared through SHM with the same generic
    copy_state_dict_to_shm/attach_state_dict_from_shm helpers every other
    workload uses - no bespoke SHM plumbing needed for quantized tensors.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float32))

    def forward(self, x):
        weight = self.qweight.to(torch.float32) * self.scale.unsqueeze(1)
        return F.linear(x, weight.to(x.dtype))


class QuantAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = QuantLinear(hidden_size, hidden_size)
        self.k_proj = QuantLinear(hidden_size, hidden_size)
        self.v_proj = QuantLinear(hidden_size, hidden_size)
        self.o_proj = QuantLinear(hidden_size, hidden_size)

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


class QuantMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = QuantLinear(hidden_size, intermediate_size)
        self.up_proj = QuantLinear(hidden_size, intermediate_size)
        self.down_proj = QuantLinear(intermediate_size, hidden_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QuantDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = QuantAttention(hidden_size, num_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = QuantMLP(hidden_size, intermediate_size)

    def forward(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        return h + self.mlp(self.post_attention_layernorm(h))


class QuantizedDenseModel(nn.Module):
    """Same LLaMA-style decoder shape as PT-Dense, but every Linear projection
    is a QuantLinear. Mirrors PT-Dense's module names 1:1 so weights can be
    quantized layer-by-layer straight out of an ordinary fp32 reference model.
    """

    def __init__(self, hidden_size=512, num_layers=4, num_heads=8, intermediate_size=1376):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            [
                QuantDecoderLayer(hidden_size, num_heads, intermediate_size)
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
        "dtype": torch.float32,
        "batch_size": 2,
        "seq_len": 16,
    }
