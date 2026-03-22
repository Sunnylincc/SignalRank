from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F


class Tower(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(ids)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_vocab: int, item_vocab: int, embedding_dim: int):
        super().__init__()
        self.user_tower = Tower(user_vocab, embedding_dim)
        self.item_tower = Tower(item_vocab, embedding_dim)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_tower(user_ids)
        i = self.item_tower(item_ids)
        return (u * i).sum(dim=-1)

    def retrieval_loss(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_tower(user_ids)
        i = self.item_tower(item_ids)
        logits = u @ i.T
        labels = torch.arange(user_ids.shape[0], device=user_ids.device)
        return F.cross_entropy(logits, labels)


@dataclass(slots=True)
class RetrievalArtifacts:
    item_ids: list[int]
    item_embeddings: list[list[float]]
