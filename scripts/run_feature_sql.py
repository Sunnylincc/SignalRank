from pathlib import Path
import duckdb


SQL_DIRS = [
    "sql/features",
    "sql/training",
    "sql/evaluation",
    "sql/diagnostics",
]


def main() -> None:
    db_path = Path("data/sample/signalrank.duckdb")
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS mart")
    con.execute("CREATE OR REPLACE TABLE interactions AS SELECT * FROM read_csv_auto('data/sample/interactions.csv')")
    con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM read_csv_auto('data/sample/items.csv')")

    for sql_dir in SQL_DIRS:
        for sql_file in sorted(Path(sql_dir).glob("*.sql")):
            con.execute(sql_file.read_text())
            print(f"applied {sql_file}")

    summary = con.execute(
        "SELECT COUNT(*) AS train_rows, (SELECT COUNT(*) FROM mart.offline_metric_slices) AS metric_rows FROM mart.training_examples"
    ).fetchone()
    print({"training_examples": summary[0], "metric_slices": summary[1]})


if __name__ == "__main__":
    main()
