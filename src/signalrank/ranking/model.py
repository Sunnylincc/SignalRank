import torch
from torch import nn


class WideAndDeepRanker(nn.Module):
    def __init__(self, user_vocab: int, item_vocab: int, surface_vocab: int, dense_dim: int, hidden_dims: list[int]):
        super().__init__()
        emb_dim = 16
        self.user_emb = nn.Embedding(user_vocab, emb_dim)
        self.item_emb = nn.Embedding(item_vocab, emb_dim)
        self.surface_emb = nn.Embedding(surface_vocab, 8)

        in_dim = emb_dim * 2 + 8 + dense_dim
        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.deep = nn.Sequential(*layers)
        self.wide = nn.Linear(dense_dim, 1)

    def forward(self, user_id: torch.Tensor, item_id: torch.Tensor, surface_id: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_emb(user_id), self.item_emb(item_id), self.surface_emb(surface_id), dense], dim=-1)
        logits = self.deep(x) + self.wide(dense)
        return logits.squeeze(-1)
