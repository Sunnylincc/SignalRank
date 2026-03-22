from pathlib import Path
import duckdb
import numpy as np

from signalrank.evaluation.metrics import auc, binary_logloss, mrr_at_k, ndcg_at_k


def main() -> None:
    con = duckdb.connect(str(Path("data/sample/signalrank.duckdb")))
    df = con.execute("SELECT label, pred_score FROM mart.scored_eval_examples").df()
    y = df["label"].to_numpy(dtype=np.float32)
    p = df["pred_score"].to_numpy(dtype=np.float32)

    print({
        "auc": round(auc(y, p), 6),
        "logloss": round(binary_logloss(y, p), 6),
        "ndcg@5": round(ndcg_at_k(y.tolist(), p.tolist(), 5), 6),
        "mrr@5": round(mrr_at_k(y.tolist(), p.tolist(), 5), 6),
    })

    print(con.execute("SELECT * FROM mart.offline_metric_slices ORDER BY rows DESC").df())


if __name__ == "__main__":
    main()
