import torch

from benchmark.pt_dense.model import LlamaDenseModel

# PT-MultiReader validates concurrent *reader* scaling, not model size, so it
# reuses PT-Dense's architecture at a smaller scale to keep N concurrent
# subprocess startups (torch import + attach) fast.
SharedBackboneModel = LlamaDenseModel


def default_config():
    return {
        "hidden_size": 256,
        "num_layers": 3,
        "num_heads": 8,
        "intermediate_size": 688,
        "dtype": torch.float16,
        "batch_size": 2,
        "seq_len": 8,
        "num_readers": 4,
    }
