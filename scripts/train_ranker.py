import torch
import torch.nn.functional as F
from signalrank.ranking.model import WideAndDeepRanker


def main() -> None:
    model = WideAndDeepRanker(user_vocab=50000, item_vocab=250000, surface_vocab=32, dense_dim=16, hidden_dims=[128, 64, 32])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(20):
        batch = 256
        logits = model(
            torch.randint(0, 50000, (batch,)),
            torch.randint(0, 250000, (batch,)),
            torch.randint(0, 32, (batch,)),
            torch.randn(batch, 16),
        )
        y = torch.randint(0, 2, (batch,), dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5 == 0:
            print(f"step={step} loss={loss.item():.4f}")
    torch.save(model.state_dict(), 'data/sample/ranker_model.pt')


if __name__ == '__main__':
    main()
