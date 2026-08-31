import torch

from benchmark.common.seeds import deterministic_seed
from benchmark.common.shm_weights import (
    attach_state_dict_from_shm,
    copy_state_dict_to_shm,
    total_weight_bytes,
)
from benchmark.pt_kvcache.model import IncrementalDecoderModel, default_config


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


def weight_state_dict(model):
    return {name: tensor.detach() for name, tensor in model.state_dict().items()}


def required_region_bytes(model):
    return total_weight_bytes(weight_state_dict(model))


def write_weights_to_shm(region, model, recorder=None):
    state_dict = weight_state_dict(model)
    copy_state_dict_to_shm(region, state_dict, recorder=recorder)
    return list(state_dict.keys())


def attach_weights_from_shm(region, model, recorder=None):
    template = model.state_dict()
    attached = attach_state_dict_from_shm(region, template, recorder=recorder)
    model.load_state_dict(attached, assign=True)
    return attached


def token_hidden_state(step, config):
    """Deterministic per-step "embedding" for the new token at decode step `step`.

    Both writer and reader call this independently and get identical tensors,
    so the only thing that can make their trajectories diverge is a bug in
    how the KV cache was grown/relocated/read through shared memory.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(deterministic_seed(f"kv_token_step_{step}"))
    return torch.randn(
        config["batch_size"], 1, config["hidden_size"], generator=generator, dtype=config["dtype"]
    )


def kv_element_size(config):
    return torch.tensor(0.0, dtype=config["dtype"]).element_size()


def kv_head_dim(config):
    return config["hidden_size"] // config["num_heads"]


def kv_shm_shape(config, length):
    """Shape of the SHM-resident cache buffer: sequence position is the
    OUTERMOST dimension. This is required for `shm_resize()`'s append-by-copy
    growth to be safe: growing any dimension other than the outermost one
    would shift every later element's byte offset (e.g. growing a middle
    "length" axis in a [batch, heads, length, dim] layout shifts every head's
    slice except head 0), silently corrupting already-written tokens.
    Assumes batch_size == 1 (asserted in default_config()'s consumers).
    """
    return [length, config["num_heads"], kv_head_dim(config)]


def kv_tensor_nbytes(config, length):
    length_val, num_heads, head_dim = kv_shm_shape(config, length)
    return length_val * num_heads * head_dim * kv_element_size(config)


def to_shm_layout(t):
    """[batch=1, heads, length, dim] (model/attention layout) -> [length, heads, dim]."""
    return t.squeeze(0).permute(1, 0, 2).contiguous()


def from_shm_layout(t):
    """[length, heads, dim] (SHM layout) -> [batch=1, heads, length, dim]."""
    return t.permute(1, 0, 2).unsqueeze(0).contiguous()


def kv_k_name(layer_idx):
    return f"kv_cache.k.{layer_idx}"


def kv_v_name(layer_idx):
    return f"kv_cache.v.{layer_idx}"


def kv_meta_name(layer_idx):
    return f"kv_cache.meta.{layer_idx}"
