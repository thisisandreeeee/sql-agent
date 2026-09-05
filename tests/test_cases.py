import sqlite3
import unittest
from pathlib import Path

from evals.runner import load_cases


ROOT = Path(__file__).resolve().parents[1]
DATABASE = ROOT / "data" / "database" / "formula_1" / "formula_1.sqlite"


class CaseSuiteTests(unittest.TestCase):
    def test_case_groups_have_the_target_counts(self):
        self.assertEqual(len(load_cases(group="basic")), 100)
        self.assertEqual(len(load_cases(group="advanced")), 100)
        self.assertEqual(len(load_cases(group="behavioral")), 50)
        self.assertEqual(len(load_cases()), 250)

    def test_sql_cases_are_read_only_and_executable(self):
        database = DATABASE.resolve()
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            for case in load_cases():
                if not case.sql_required:
                    self.assertIsNone(case.gold_sql, case.name)
                    continue

                self.assertTrue(case.gold_sql, case.name)
                sql = case.gold_sql.strip()
                self.assertTrue(
                    sql.upper().startswith(("SELECT", "WITH")),
                    case.name,
                )
                self.assertNotIn(";", sql.rstrip(";"), case.name)
                rows = connection.execute(sql).fetchall()
                if case.name.startswith(("basic_", "advanced_")):
                    self.assertTrue(rows, case.name)


if __name__ == "__main__":
    unittest.main()
