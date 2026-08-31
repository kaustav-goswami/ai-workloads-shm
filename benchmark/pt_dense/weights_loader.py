import torch

from benchmark.common.seeds import deterministic_seed
from benchmark.common.shm_weights import (
    attach_state_dict_from_shm,
    copy_state_dict_to_shm,
    total_weight_bytes,
)
from benchmark.pt_dense.model import LlamaDenseModel, default_config


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


def weight_state_dict(model):
    return {name: tensor.detach() for name, tensor in model.state_dict().items()}


def weight_names(model):
    return list(model.state_dict().keys())


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


def reference_forward(model, x):
    with torch.no_grad():
        return model(x)
