import torch

from benchmark.common.seeds import deterministic_seed
from benchmark.common.shm_weights import (
    attach_state_dict_from_shm,
    copy_state_dict_to_shm,
    total_weight_bytes,
)
from benchmark.pt_vision.model import SimpleCNN, default_config


def build_model(config=None):
    config = default_config() if config is None else config
    model = SimpleCNN(
        in_channels=config["in_channels"],
        stem_channels=config["stem_channels"],
        num_blocks=config["num_blocks"],
        num_classes=config["num_classes"],
    )
    return model.to(config["dtype"])


def initialize_weights(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(deterministic_seed(name))
            if name.endswith(".bias"):
                param.zero_()
            else:
                fan_in = param.shape[1:].numel()
                bound = (1.0 / fan_in) ** 0.5
                param.copy_(
                    (torch.rand(param.shape, generator=generator, dtype=param.dtype) * 2 - 1) * bound
                )


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


def local_image_batch(config=None):
    config = default_config() if config is None else config
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    return torch.randn(
        config["batch_size"],
        config["in_channels"],
        config["image_size"],
        config["image_size"],
        generator=generator,
        dtype=config["dtype"],
    )


def reference_forward(model, x):
    with torch.no_grad():
        return model(x)
