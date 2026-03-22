import duckdb
from pathlib import Path


def run_sql_dir(con: duckdb.DuckDBPyConnection, sql_dir: Path) -> None:
    for sql_file in sorted(sql_dir.glob('*.sql')):
        con.execute(sql_file.read_text())
