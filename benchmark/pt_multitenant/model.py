import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark.common.seeds import deterministic_seed
from benchmark.common.timing import TimingRecorder
from benchmark.pt_dense.model import LlamaDecoderLayer, RMSNorm


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)

    def forward(self, x):
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling


class MoEExpert(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoELayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts=8, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MoEExpert(hidden_size, intermediate_size) for _ in range(num_experts)]
        )

    def forward(self, x):
        batch, seq_len, hidden = x.shape
        flat = x.view(-1, hidden)
        logits = self.gate(flat)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            mask = (indices == expert_id).any(dim=-1)
            if not mask.any():
                continue
            token_idx = mask.nonzero(as_tuple=False).squeeze(-1)
            selected = flat[token_idx]
            expert_out = expert(selected)
            for k in range(self.top_k):
                route_mask = indices[token_idx, k] == expert_id
                if not route_mask.any():
                    continue
                route_idx = token_idx[route_mask]
                scale = weights[route_idx, k].unsqueeze(-1)
                out[route_idx] += expert_out[route_mask] * scale
        return out.view(batch, seq_len, hidden)


class MultiTenantLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, num_experts=8, lora_rank=16):
        super().__init__()
        base_layer = LlamaDecoderLayer(hidden_size, num_heads, intermediate_size)
        self.input_layernorm = base_layer.input_layernorm
        attn = base_layer.self_attn
        self.q_proj = LoRALinear(attn.q_proj, rank=lora_rank)
        self.k_proj = attn.k_proj
        self.v_proj = LoRALinear(attn.v_proj, rank=lora_rank)
        self.o_proj = attn.o_proj
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        self.moe = MoELayer(hidden_size, intermediate_size, num_experts=num_experts)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        normed = self.input_layernorm(x)
        q = self.q_proj(normed).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        h = x + self.o_proj(context)
        return h + self.moe(self.post_attention_layernorm(h))


class MultiTenantModel(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_layers=2,
        num_heads=8,
        intermediate_size=1376,
        num_experts=8,
        lora_rank=16,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MultiTenantLayer(
                    hidden_size,
                    num_heads,
                    intermediate_size,
                    num_experts=num_experts,
                    lora_rank=lora_rank,
                )
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
        "num_layers": 2,
        "num_heads": 8,
        "intermediate_size": 1376,
        "num_experts": 8,
        "lora_rank": 16,
        "lora_alpha": 32,
        "dtype": torch.float16,
        "batch_size": 2,
        "seq_len": 8,
    }


def build_model(config=None):
    config = default_config() if config is None else config
    model = MultiTenantModel(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        intermediate_size=config["intermediate_size"],
        num_experts=config["num_experts"],
        lora_rank=config["lora_rank"],
    )
    return model.to(config["dtype"])


def initialize_weights(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(deterministic_seed(name))
            if param.ndim == 1:
                param.fill_(1.0)
            elif "lora_a" in name:
                param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.01)
            elif "lora_b" in name:
                param.zero_()
            elif "gate" in name and "moe" in name:
                param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.01)
            else:
                param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.02)


def local_activation_input(config=None):
    config = default_config() if config is None else config
    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    return torch.randn(
        config["batch_size"],
        config["seq_len"],
        config["hidden_size"],
        generator=generator,
        dtype=config["dtype"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="PT-MultiTenant: shared LLaMA backbone with per-layer LoRA adapters and "
        "an 8-expert Top-2 MoE. All weights are ordinary tensors allocated with PyTorch's "
        "default CPU allocator."
    )
    parser.parse_args()

    print("PT-MultiTenant: building hybrid model and initializing weights...")
    config = default_config()
    recorder = TimingRecorder(label="PT-MultiTenant")
    with recorder.time("build"):
        model = build_model(config)
    with recorder.time("init"):
        initialize_weights(model)

    x = local_activation_input(config)
    print(f"PT-MultiTenant: running forward pass on activations {tuple(x.shape)}...")
    with recorder.time("forward"):
        with torch.no_grad():
            out = model(x)
    recorder.print_summary()

    print(f"PT-MultiTenant: output shape {tuple(out.shape)}, mean {out.float().mean().item():.6f}")
    print("PT-MultiTenant: SUCCESS - routed adapter/MoE forward pass completed.")


if __name__ == "__main__":
    main()
