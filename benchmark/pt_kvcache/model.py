import argparse
import math

import torch
import torch.nn.functional as F

from benchmark.common.seeds import deterministic_seed
from benchmark.common.timing import TimingRecorder
from benchmark.pt_dense.model import LlamaAttention, LlamaMLP, RMSNorm


class IncrementalDecoderLayer(torch.nn.Module):
    """A decoder layer that decodes one token at a time against a per-layer
    K/V cache. The cache is an ordinary tensor that grows via `torch.cat` on
    every step - PyTorch's default allocator handles the reallocation, so
    there is no manual resize/relocate bookkeeping to do.
    """

    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = LlamaAttention(hidden_size, num_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = LlamaMLP(hidden_size, intermediate_size)

    def _project_qkv(self, x_t):
        batch = x_t.shape[0]
        attn = self.self_attn
        normed = self.input_layernorm(x_t)
        shape = (batch, 1, attn.num_heads, attn.head_dim)
        q = attn.q_proj(normed).view(*shape).transpose(1, 2)
        k = attn.k_proj(normed).view(*shape).transpose(1, 2)
        v = attn.v_proj(normed).view(*shape).transpose(1, 2)
        return q, k, v

    def _finish(self, x_t, ctx):
        batch = x_t.shape[0]
        ctx = ctx.transpose(1, 2).contiguous().view(batch, 1, -1)
        h = x_t + self.self_attn.o_proj(ctx)
        return h + self.mlp(self.post_attention_layernorm(h))

    def step(self, x_t, k_cache, v_cache):
        """Compute this token's k/v, append to the cache, and attend over the
        full grown cache. Returns (output, new_k_cache, new_v_cache)."""
        q, new_k, new_v = self._project_qkv(x_t)
        k_all = new_k if k_cache is None else torch.cat([k_cache, new_k], dim=2)
        v_all = new_v if v_cache is None else torch.cat([v_cache, new_v], dim=2)
        scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.self_attn.head_dim)
        ctx = torch.matmul(F.softmax(scores, dim=-1), v_all)
        return self._finish(x_t, ctx), k_all, v_all


class IncrementalDecoderModel(torch.nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, num_heads=4, intermediate_size=176):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layers = torch.nn.ModuleList(
            [
                IncrementalDecoderLayer(hidden_size, num_heads, intermediate_size)
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size)

    def decode_step(self, x_t, kv_caches):
        """Decode one new token across every layer.

        `kv_caches` is a list of (k, v) tuples (or (None, None) at step 0),
        one per layer, holding the cache up to (not including) this token.
        Returns (final_hidden, new_kv_caches) where new_kv_caches[l] includes
        this token appended.
        """
        x = x_t
        new_caches = []
        for layer, (k_cache, v_cache) in zip(self.layers, kv_caches):
            x, k_all, v_all = layer.step(x, k_cache, v_cache)
            new_caches.append((k_all, v_all))
        return self.norm(x), new_caches


def default_config():
    return {
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "intermediate_size": 176,
        "dtype": torch.float32,
        "batch_size": 1,
        "max_new_tokens": 24,
    }


def build_model(config=None):
    config = default_config() if config is None else config
    model = IncrementalDecoderModel(
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
            if param.ndim == 1:
                param.fill_(1.0)
            else:
                param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.02)


def token_hidden_state(step, config):
    """Deterministic per-step "embedding" for the new token at decode step `step`."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(deterministic_seed(f"kv_token_step_{step}"))
    return torch.randn(
        config["batch_size"], 1, config["hidden_size"], generator=generator, dtype=config["dtype"]
    )


def main():
    parser = argparse.ArgumentParser(
        description="PT-KVCache: incremental decode over a per-layer K/V cache that grows "
        "one token at a time. The cache is an ordinary PyTorch tensor grown with torch.cat, "
        "using PyTorch's default CPU allocator - no manual shared-memory resize is needed."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the number of decode steps from default_config().",
    )
    args = parser.parse_args()

    config = default_config()
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens

    print("PT-KVCache: building incremental decoder and initializing weights...")
    model = build_model(config)
    initialize_weights(model)

    num_layers = config["num_layers"]
    recorder = TimingRecorder(label="PT-KVCache decode")
    kv_caches = [(None, None) for _ in range(num_layers)]
    last_out = None
    for step in range(config["max_new_tokens"]):
        x = token_hidden_state(step, config)
        with recorder.time("step"):
            with torch.no_grad():
                last_out, kv_caches = model.decode_step(x, kv_caches)

    print(f"PT-KVCache: completed {config['max_new_tokens']} decode steps across {num_layers} layers.")
    recorder.print_summary()

    cache_len = kv_caches[0][0].shape[2]
    print(
        f"PT-KVCache: final cache length {cache_len}, "
        f"last output mean {last_out.float().mean().item():.6f}"
    )
    print("PT-KVCache: SUCCESS - incremental decode with a growing KV cache completed.")


if __name__ == "__main__":
    main()
