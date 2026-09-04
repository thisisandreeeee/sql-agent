"""Create the disposable runtime database from Spider's official fixture."""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "database" / "formula_1" / "formula_1.sqlite"
DEFAULT_TARGET = ROOT / "var" / "sql-agent.sqlite"
REQUIRED_TABLES = {
    "circuits",
    "constructorResults",
    "constructorStandings",
    "constructors",
    "driverStandings",
    "drivers",
    "lapTimes",
    "pitStops",
    "qualifying",
    "races",
    "results",
    "seasons",
    "status",
}


def open_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)


def validate_source(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"SQLite source does not exist: {path}")

    with open_read_only(path) as connection:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {path}")

        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]

    missing = REQUIRED_TABLES.difference(tables)
    if missing:
        raise RuntimeError(f"SQLite source is missing tables: {sorted(missing)}")
    return tables


def seed(source: Path, target: Path) -> None:
    source = source.resolve()
    target = target.resolve()
    if source == target:
        raise ValueError("Source and target must be different files")

    tables = validate_source(source)
    target.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"Seeded {target} ({len(tables)} tables)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    seed(args.source, args.target)


if __name__ == "__main__":
    main()
