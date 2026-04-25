from pathlib import Path
import duckdb

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cricket.duckdb"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
VIEWS_PATH = Path(__file__).resolve().parent / "views.sql"


def connect(db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(SCHEMA_PATH.read_text())
    return con


def install_views(db_path: Path | str | None = None) -> None:
    """Create or refresh derived analytical views. Safe to call repeatedly."""
    con = connect(db_path)
    con.execute(VIEWS_PATH.read_text())
    con.close()
