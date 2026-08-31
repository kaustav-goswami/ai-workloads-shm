import math

import torch
import torch.nn.functional as F

from benchmark.pt_dense.model import LlamaAttention, LlamaMLP, RMSNorm


class IncrementalDecoderLayer(torch.nn.Module):
    """A decoder layer that supports two forward modes over an external KV cache:

    - `step()`: the writer's mode. Computes this token's k/v from scratch,
      appends them to the given cache, and attends. Returns the grown cache
      so the caller can persist the new tail slice into shared memory.
    - `attend_with_cache()`: the reader's mode. Computes only the query
      locally and attends against a full k/v cache handed in verbatim (as
      read back from shared memory), so a bug in the shared growing buffer
      shows up as a numerical mismatch against the writer's own trajectory.
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
        """Writer path: compute + append this token's k/v, then attend."""
        q, new_k, new_v = self._project_qkv(x_t)
        k_all = new_k if k_cache is None else torch.cat([k_cache, new_k], dim=2)
        v_all = new_v if v_cache is None else torch.cat([v_cache, new_v], dim=2)
        scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.self_attn.head_dim)
        ctx = torch.matmul(F.softmax(scores, dim=-1), v_all)
        return self._finish(x_t, ctx), k_all, v_all

    def attend_with_cache(self, x_t, k_all, v_all):
        """Reader path: compute q locally, attend against a cache read verbatim
        from shared memory (no local recomputation of k/v)."""
        q, _, _ = self._project_qkv(x_t)
        scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(self.self_attn.head_dim)
        ctx = torch.matmul(F.softmax(scores, dim=-1), v_all)
        return self._finish(x_t, ctx)


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
        """Writer path for one new token across every layer.

        `kv_caches` is a list of (k, v) tuples (or (None, None) at step 0),
        one per layer, holding the full cache up to (not including) this
        token. Returns (final_hidden, new_kv_caches) where new_kv_caches[l]
        includes this token appended.
        """
        x = x_t
        new_caches = []
        for layer, (k_cache, v_cache) in zip(self.layers, kv_caches):
            x, k_all, v_all = layer.step(x, k_cache, v_cache)
            new_caches.append((k_all, v_all))
        return self.norm(x), new_caches

    def decode_step_from_cache(self, x_t, kv_caches):
        """Reader path for one new token: attend against caches read from SHM."""
        x = x_t
        for layer, (k_all, v_all) in zip(self.layers, kv_caches):
            x = layer.attend_with_cache(x, k_all, v_all)
        return self.norm(x)


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
