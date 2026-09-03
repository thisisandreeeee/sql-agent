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

            self.assertEqual(result, "Error: attempt to write a readonly database")
            self.assertEqual(db.query("SELECT COUNT(*) FROM records", database), "[(1,)]")


if __name__ == "__main__":
    unittest.main()
