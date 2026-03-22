import torch
from signalrank.retrieval.two_tower import TwoTowerModel


def main() -> None:
    model = TwoTowerModel(user_vocab=50000, item_vocab=250000, embedding_dim=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(20):
        user = torch.randint(0, 50000, (128,))
        item = torch.randint(0, 250000, (128,))
        loss = model.retrieval_loss(user, item)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5 == 0:
            print(f"step={step} loss={loss.item():.4f}")
    torch.save(model.state_dict(), 'data/sample/retrieval_model.pt')


if __name__ == '__main__':
    main()
