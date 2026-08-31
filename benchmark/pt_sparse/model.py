import torch
import torch.nn as nn


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
