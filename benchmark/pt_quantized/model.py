import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.common.timing import TimingRecorder
from benchmark.pt_dense.model import LlamaDenseModel, RMSNorm
from benchmark.pt_dense.model import initialize_weights as initialize_reference_weights
from benchmark.pt_dense.model import local_activation_input


class QuantLinear(nn.Module):
    """A Linear layer whose weight lives as int8 + a per-output-channel fp32
    scale (symmetric quantization, no zero-point), dequantized on the fly in
    forward(). qweight/scale are ordinary buffers allocated with PyTorch's
    default CPU allocator.
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


def build_reference_model(config=None):
    """The 'true' fp32 model that gets quantized. Reuses PT-Dense's exact
    architecture and deterministic initialization.
    """
    config = default_config() if config is None else config
    model = LlamaDenseModel(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        intermediate_size=config["intermediate_size"],
    )
    initialize_reference_weights(model)
    return model


def build_quantized_model(config=None):
    config = default_config() if config is None else config
    return QuantizedDenseModel(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        intermediate_size=config["intermediate_size"],
    )


def quantize_tensor(weight):
    """Per-output-channel symmetric int8 quantization (no zero-point):
    scale[o] = max(|weight[o, :]|) / 127, qweight[o, i] = round(weight[o, i] / scale[o]).
    """
    absmax = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = absmax / 127.0
    qweight = torch.clamp(torch.round(weight / scale), -127, 127).to(torch.int8)
    return qweight, scale.squeeze(1)


def _quantize_into(quant_linear, fp32_weight):
    qweight, scale = quantize_tensor(fp32_weight)
    with torch.no_grad():
        quant_linear.qweight.copy_(qweight)
        quant_linear.scale.copy_(scale)


def quantize_model(quant_model, ref_model):
    """Populate every QuantLinear buffer in `quant_model` by quantizing the
    matching fp32 weight from `ref_model`. Both models share identical
    submodule names (mirrored 1:1), so layers line up positionally.
    """
    with torch.no_grad():
        quant_model.norm.weight.copy_(ref_model.norm.weight)
        for qlayer, rlayer in zip(quant_model.layers, ref_model.layers):
            qlayer.input_layernorm.weight.copy_(rlayer.input_layernorm.weight)
            qlayer.post_attention_layernorm.weight.copy_(rlayer.post_attention_layernorm.weight)
            _quantize_into(qlayer.self_attn.q_proj, rlayer.self_attn.q_proj.weight)
            _quantize_into(qlayer.self_attn.k_proj, rlayer.self_attn.k_proj.weight)
            _quantize_into(qlayer.self_attn.v_proj, rlayer.self_attn.v_proj.weight)
            _quantize_into(qlayer.self_attn.o_proj, rlayer.self_attn.o_proj.weight)
            _quantize_into(qlayer.mlp.gate_proj, rlayer.mlp.gate_proj.weight)
            _quantize_into(qlayer.mlp.up_proj, rlayer.mlp.up_proj.weight)
            _quantize_into(qlayer.mlp.down_proj, rlayer.mlp.down_proj.weight)
    return quant_model


def main():
    parser = argparse.ArgumentParser(
        description="PT-Quantized: quantize an fp32 LLaMA-style decoder to per-channel "
        "int8 weight + fp32 scale. Both the fp32 reference and the quantized model live "
        "in ordinary process memory allocated by PyTorch's default CPU allocator."
    )
    parser.parse_args()

    print("PT-Quantized: building fp32 reference model and quantizing to int8...")
    config = default_config()
    ref_model = build_reference_model(config)
    quant_model = build_quantized_model(config)
    quantize_model(quant_model, ref_model)

    fp32_bytes = sum(t.numel() * t.element_size() for t in ref_model.state_dict().values())
    quant_bytes = sum(t.numel() * t.element_size() for t in quant_model.state_dict().values())
    print(
        f"PT-Quantized: fp32 footprint={fp32_bytes / (1024 * 1024):.2f} MiB, "
        f"quantized footprint={quant_bytes / (1024 * 1024):.2f} MiB "
        f"({fp32_bytes / quant_bytes:.2f}x smaller)."
    )

    x = local_activation_input(config)
    recorder = TimingRecorder(label="PT-Quantized")
    print(f"PT-Quantized: running forward pass with activations {tuple(x.shape)}...")
    with recorder.time("forward"):
        with torch.no_grad():
            out = quant_model(x)
    recorder.print_summary()

    with torch.no_grad():
        fp32_out = ref_model(x)
    quant_error = (out - fp32_out).abs().max().item()
    rel_error = quant_error / fp32_out.abs().max().item()
    print(
        f"PT-Quantized: int8 quantization error vs true fp32 - "
        f"max abs diff {quant_error:.4f} ({rel_error * 100:.2f}% of peak fp32 magnitude)."
    )
    print("PT-Quantized: SUCCESS - quantized forward pass completed.")


if __name__ == "__main__":
    main()
