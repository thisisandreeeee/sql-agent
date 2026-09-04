"""Run one representative query against the seeded Spider Formula 1 database."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE = ROOT / "var" / "sql-agent.sqlite"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    args = parser.parse_args()

    if not args.database.is_file():
        raise SystemExit(f"Database does not exist; run seed_db.py first: {args.database}")

    connection = sqlite3.connect(
        f"file:{args.database.resolve()}?mode=ro", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise AssertionError("SQLite integrity check failed")

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        missing = REQUIRED_TABLES - tables
        if missing:
            raise AssertionError(f"Missing expected tables: {sorted(missing)}")

        rows = connection.execute(
            """
            SELECT
                d.forename,
                d.surname,
                COUNT(*) AS wins
            FROM results AS r
            JOIN drivers AS d ON d.driverId = r.driverId
            WHERE r.position = '1'
            GROUP BY d.driverId, d.forename, d.surname
            ORDER BY wins DESC
            LIMIT 10
            """
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise AssertionError("Smoke query returned no rows")

    for row in rows:
        print(f"{row['forename']} {row['surname']}: {row['wins']} wins")


if __name__ == "__main__":
    main()
