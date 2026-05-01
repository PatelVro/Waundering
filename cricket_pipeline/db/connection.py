import time
from pathlib import Path
import duckdb

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cricket.duckdb"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
VIEWS_PATH = Path(__file__).resolve().parent / "views.sql"

# Default retry policy for the file-level lock contention that DuckDB throws
# when a predict_match subprocess (read-write, exclusive) is mid-feature-build
# and the orchestrator's bet_engine / export tries to open the same file.
# Total wait time at default settings: 0.4s × (1.4 ** 0..11) ≈ ~30s, which
# comfortably covers a fast-mode predict_match cycle.
_LOCK_RETRIES      = 12
_LOCK_BASE_DELAY_S = 0.4
_LOCK_BACKOFF      = 1.4


def _is_lock_error(exc: BaseException) -> bool:
    """Recognise DuckDB's "exclusive file lock held by another process" error.
    On Windows the message reads 'The process cannot access the file because
    it is being used by another process.'; cross-platform we also accept the
    generic 'lock' / 'in use' phrasing in case the wording shifts."""
    msg = str(exc).lower()
    return any(s in msg for s in (
        "process cannot access",
        "being used by another",
        "could not set lock",
        "could not obtain lock",
        "lock could not be set",
    ))


def connect(db_path: Path | str | None = None,
             retries: int = _LOCK_RETRIES,
             read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the cricket DuckDB. Retries on file-lock contention with
    exponential backoff so that brief overlap with a predict_match
    subprocess doesn't kill bet-engine / export operations.

    `retries=0` disables retry (fail-fast). `read_only=True` opens a
    read-only handle, which DuckDB allows alongside an exclusive writer.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    last_err = None
    for attempt in range(retries + 1):
        try:
            con = duckdb.connect(str(path), read_only=read_only)
            # Skip schema setup on read-only handles (no DDL allowed)
            if not read_only:
                con.execute(SCHEMA_PATH.read_text())
            return con
        except (duckdb.IOException, OSError) as e:
            last_err = e
            if not _is_lock_error(e) or attempt == retries:
                raise
            # Exponential backoff
            time.sleep(_LOCK_BASE_DELAY_S * (_LOCK_BACKOFF ** attempt))
    raise last_err  # unreachable but keeps the type checker quiet


def install_views(db_path: Path | str | None = None) -> None:
    """Create or refresh derived analytical views. Safe to call repeatedly."""
    con = connect(db_path)
    con.execute(VIEWS_PATH.read_text())
    con.close()
