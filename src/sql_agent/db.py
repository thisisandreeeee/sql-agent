"""SQLite utilities used by the SQL agent tools."""

import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE = ROOT / "var" / "sql-agent.sqlite"

_READ_ONLY_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def _read_only_authorizer(action, arg1, arg2, database_name, trigger_name):
    """Allow queries only; deny writes, attachments, and connection changes."""
    return (
        sqlite3.SQLITE_OK
        if action in _READ_ONLY_ACTIONS
        else sqlite3.SQLITE_DENY
    )


def _connect_read_only(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    connection.set_authorizer(_read_only_authorizer)
    return connection


def list_tables(database: Path = DEFAULT_DATABASE) -> list[str]:
    """Return non-system table names from the SQLite database."""
    with _connect_read_only(database) as connection:
        return [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]


def schema(table_names: str, database: Path = DEFAULT_DATABASE) -> str:
    """Return table definitions and up to three sample rows."""
    names = [name.strip() for name in table_names.split(",") if name.strip()]
    if not names:
        return "Error: at least one table name is required"

    with _connect_read_only(database) as connection:
        definitions = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND name IN "
                f"({','.join('?' for _ in names)})",
                names,
            )
        }
        missing = [name for name in names if name not in definitions]
        if missing:
            return f"Error: unknown table(s): {', '.join(missing)}"

        results = []
        for name in names:
            identifier = '"' + name.replace('"', '""') + '"'
            cursor = connection.execute(f"SELECT * FROM {identifier} LIMIT 3")
            columns = [column[0] for column in cursor.description or []]
            rows = cursor.fetchall()
            sample = "\n".join(
                "\t".join(str(value) for value in row) for row in rows
            )
            results.append(
                f"{definitions[name]}\n\n"
                f"/*\n3 rows from {name} table:\n"
                f"{'\t'.join(columns)}\n{sample}\n*/"
            )
        return "\n\n".join(results)


def query(sql: str, database: Path = DEFAULT_DATABASE) -> str:
    """Execute a read-only SQL query and return its rows as text."""
    if not sql.strip():
        return "Error: SQL query is required"

    with _connect_read_only(database) as connection:
        try:
            rows = connection.execute(sql).fetchall()
        except sqlite3.Error as error:
            return f"Error: {error}"
    return str(rows)
