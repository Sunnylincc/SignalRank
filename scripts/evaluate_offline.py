import numpy as np
from signalrank.evaluation.metrics import auc, binary_logloss, mrr_at_k, ndcg_at_k


def main() -> None:
    y = np.array([1, 0, 1, 0, 1, 0])
    p = np.array([0.8, 0.1, 0.7, 0.3, 0.6, 0.2])
    print('AUC', auc(y, p))
    print('LogLoss', binary_logloss(y, p))
    print('NDCG@5', ndcg_at_k(y.tolist(), p.tolist(), 5))
    print('MRR@5', mrr_at_k(y.tolist(), p.tolist(), 5))


if __name__ == '__main__':
    main()
