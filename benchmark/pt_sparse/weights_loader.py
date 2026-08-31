import torch

from benchmark.common.seeds import deterministic_seed
from benchmark.common.shm_weights import (
    attach_state_dict_from_shm,
    copy_state_dict_to_shm,
    total_weight_bytes,
)
from benchmark.pt_sparse.model import DLRMModel, default_config


def build_model(config=None):
    config = default_config() if config is None else config
    model = DLRMModel(
        num_tables=config["num_tables"],
        num_embeddings=config["num_embeddings"],
        embedding_dim=config["embedding_dim"],
        bottom_mlp_dims=config["bottom_mlp_dims"],
    )
    return model.to(config["dtype"])


def initialize_weights(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(deterministic_seed(name))
            if param.ndim < 2:
                param.fill_(0.01)
            elif "tables" in name:
                param.copy_(torch.randn(param.shape, generator=generator, dtype=param.dtype) * 0.01)
            else:
                bound = (6.0 / (param.shape[0] + param.shape[1])) ** 0.5
                param.copy_(
                    (torch.rand(param.shape, generator=generator, dtype=param.dtype) * 2 - 1)
                    * bound
                )


def weight_state_dict(model):
    return {name: tensor.detach() for name, tensor in model.state_dict().items()}


def shared_weight_names(model):
    return [name for name in model.state_dict() if name.startswith("tables.")]


def local_weight_names(model):
    return [name for name in model.state_dict() if not name.startswith("tables.")]


def required_region_bytes(model):
    state_dict = weight_state_dict(model)
    shared = {name: state_dict[name] for name in shared_weight_names(model)}
    return total_weight_bytes(shared)


def write_weights_to_shm(region, model, recorder=None):
    state_dict = weight_state_dict(model)
    shared = {name: state_dict[name] for name in shared_weight_names(model)}
    copy_state_dict_to_shm(region, shared, recorder=recorder)
    return {
        "shared": list(shared.keys()),
        "local": local_weight_names(model),
    }


def attach_weights_from_shm(region, model, recorder=None):
    template = model.state_dict()
    shared_template = {name: template[name] for name in shared_weight_names(model)}
    attached_shared = attach_state_dict_from_shm(region, shared_template, recorder=recorder)
    merged = dict(template)
    merged.update(attached_shared)
    model.load_state_dict(merged, assign=True)
    return attached_shared


def local_batch(config=None):
    config = default_config() if config is None else config
    generator = torch.Generator(device="cpu")
    generator.manual_seed(99)
    dense = torch.randn(
        config["batch_size"],
        config["bottom_mlp_dims"][0],
        generator=generator,
        dtype=config["dtype"],
    )
    sparse_indices = []
    sparse_offsets = []
    for _ in range(config["num_tables"]):
        idx = torch.randint(
            0,
            config["num_embeddings"],
            (config["batch_size"] * config["pooling_len"],),
            generator=generator,
        )
        offsets = torch.arange(
            0,
            config["batch_size"] * config["pooling_len"],
            config["pooling_len"],
            dtype=torch.int64,
        )
        sparse_indices.append(idx)
        sparse_offsets.append(offsets)
    return dense, sparse_indices, sparse_offsets


def reference_forward(model, dense, sparse_indices, sparse_offsets):
    with torch.no_grad():
        return model(dense, sparse_indices, sparse_offsets)
