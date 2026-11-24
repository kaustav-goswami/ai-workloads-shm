import argparse

import torch
import torch.nn as nn

from benchmark.common.seeds import deterministic_seed
from benchmark.common.timing import TimingRecorder


class DLRMModel(nn.Module):
    def __init__(
        self,
        num_tables=8,
        num_embeddings=10_000,
        embedding_dim=128,
        bottom_mlp_dims=(512, 256, 128),
    ):
        super().__init__()
        self.num_tables = num_tables
        self.embedding_dim = embedding_dim
        self.tables = nn.ModuleList(
            [nn.Embedding(num_embeddings, embedding_dim) for _ in range(num_tables)]
        )
        layers = []
        in_dim = bottom_mlp_dims[0]
        for hidden_dim in bottom_mlp_dims[1:]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.bottom_mlp = nn.Sequential(*layers)
        self.top_input_dim = bottom_mlp_dims[-1] + num_tables * embedding_dim
        self.top_mlp = nn.Sequential(
            nn.Linear(self.top_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, dense_features, sparse_indices, sparse_offsets):
        dense_out = self.bottom_mlp(dense_features)
        pooled = []
        for table_id, table in enumerate(self.tables):
            bag = nn.functional.embedding_bag(
                sparse_indices[table_id],
                table.weight,
                sparse_offsets[table_id],
                mode="sum",
            )
            pooled.append(bag)
        interaction = torch.cat([dense_out] + pooled, dim=1)
        return self.top_mlp(interaction)


def default_config():
    return {
        "num_tables": 8,
        "num_embeddings": 10_000,
        "embedding_dim": 128,
        "bottom_mlp_dims": (512, 256, 128),
        "batch_size": 32,
        "pooling_len": 20,
        "dtype": torch.float32,
    }


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


def main():
    parser = argparse.ArgumentParser(
        description="PT-Sparse: DLRM-style embedding tables plus bottom/top MLP. All "
        "weights are ordinary tensors allocated with PyTorch's default CPU allocator; "
        "embedding_bag performs normal in-process row lookups."
    )
    parser.parse_args()

    print("PT-Sparse: building DLRM model and initializing weights...")
    config = default_config()
    recorder = TimingRecorder(label="PT-Sparse")
    with recorder.time("build"):
        model = build_model(config)
    with recorder.time("init"):
        initialize_weights(model)

    dense, sparse_indices, sparse_offsets = local_batch(config)
    print(f"PT-Sparse: running forward pass on batch_size={config['batch_size']}...")
    with recorder.time("forward"):
        with torch.no_grad():
            out = model(dense, sparse_indices, sparse_offsets)
    recorder.print_summary()

    print(f"PT-Sparse: output shape {tuple(out.shape)}, mean {out.float().mean().item():.6f}")
    print("PT-Sparse: SUCCESS - EmbeddingBag sparse lookup forward pass completed.")


if __name__ == "__main__":
    main()
