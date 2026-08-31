import torch

from benchmark.common.shm_weights import (
    attach_state_dict_from_shm,
    copy_state_dict_to_shm,
    total_weight_bytes,
)
from benchmark.pt_dense.model import LlamaDenseModel
from benchmark.pt_dense.weights_loader import initialize_weights as initialize_reference_weights
from benchmark.pt_quantized.model import QuantizedDenseModel, default_config


def build_reference_model(config=None):
    """The 'true' fp32 model that gets quantized. Reuses PT-Dense's exact
    architecture and deterministic initialization so both the writer and the
    reader can reconstruct byte-identical fp32 weights independently.
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
