from pathlib import Path
import duckdb


def main() -> None:
    con = duckdb.connect('data/sample/signalrank.duckdb')
    con.execute("CREATE SCHEMA IF NOT EXISTS mart")
    con.execute("CREATE OR REPLACE TABLE interactions AS SELECT * FROM read_csv_auto('data/sample/interactions.csv')")
    con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM read_csv_auto('data/sample/items.csv')")
    for d in ['sql/features', 'sql/training', 'sql/evaluation', 'sql/diagnostics']:
        for file in sorted(Path(d).glob('*.sql')):
            con.execute(file.read_text())
            print(f"applied {file}")


if __name__ == '__main__':
    main()
