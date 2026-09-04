import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sql_agent import db


class DatabaseAccessTests(unittest.TestCase):
    def test_query_connection_cannot_write(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "test.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")
                connection.execute("INSERT INTO records VALUES (1)")

            result = db.query("DELETE FROM records", database)

            self.assertEqual(result, "Error: not authorized")
            self.assertEqual(db.query("SELECT COUNT(*) FROM records", database), "[(1,)]")

    def test_query_rejects_attached_databases(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "test.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE records (id INTEGER PRIMARY KEY)")

            result = db.query(
                f"ATTACH DATABASE '{directory}/other.sqlite' AS other",
                database,
            )

            self.assertEqual(result, "Error: not authorized")

    def test_query_allows_read_only_ctes(self):
        with TemporaryDirectory() as directory:
            database = Path(directory) / "test.sqlite"
            with sqlite3.connect(database) as connection:
                connection.execute("CREATE TABLE records (id INTEGER)")
                connection.execute("INSERT INTO records VALUES (1)")

            self.assertEqual(
                db.query(
                    "WITH selected AS (SELECT id FROM records) "
                    "SELECT COUNT(*) FROM selected",
                    database,
                ),
                "[(1,)]",
            )


if __name__ == "__main__":
    unittest.main()
