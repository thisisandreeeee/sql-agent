import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.runner import next_run_path, persisted_result, summarize
from unittest.mock import patch

from evals.evaluators import (
    sql_failure_evaluator,
    sql_result_evaluator,
    sql_usage_evaluator,
    sql_validity_evaluator,
)
from evals.types import EvalCase
from sql_agent.types import RunMetrics, RunResult, SqlAttempt


class SummaryTests(unittest.TestCase):
    def test_summarizes_cases_and_scores(self):
        cases = [
            {
                "result": {
                    "run_metrics": {
                        "latency_sec": 1.2,
                        "model_time_sec": 2.0,
                        "output_tokens": 10,
                        "input_tokens": 100,
                        "total_tokens": 110,
                        "sql_attempt_count": 1,
                        "sql_failed_count": 0,
                    }
                },
                "evaluations": [
                    {"key": "correctness", "score": True},
                    {"key": "relevance", "score": 0.5},
                ],
                "error": None,
            },
            {
                "result": {
                    "run_metrics": {
                        "latency_sec": 2.8,
                        "model_time_sec": 3.0,
                        "output_tokens": 20,
                        "input_tokens": 200,
                        "total_tokens": 220,
                        "sql_attempt_count": 2,
                        "sql_failed_count": 0,
                    }
                },
                "evaluations": [
                    {"key": "correctness", "score": False},
                    {"key": "relevance", "score": True},
                ],
                "error": None,
            },
        ]

        summary = summarize(cases)

        self.assertEqual(summary["total_cases"], 2)
        self.assertEqual(summary["passed_cases"], 1)
        self.assertEqual(summary["latency_sec"], {"total": 4.0, "mean": 2.0})
        self.assertEqual(summary["output_tokens"], {"total": 30, "mean": 15.0})
        self.assertEqual(summary["output_tokens_per_second"], 6.0)
        self.assertEqual(summary["sql_attempt_count"], {"total": 3, "mean": 1.5})
        self.assertEqual(summary["sql_failed_count"], {"total": 0, "mean": 0.0})
        self.assertEqual(
            summary["score_means"], {"correctness": 0.5, "relevance": 0.75}
        )

    def test_persisted_result_excludes_messages_and_duplicate_question(self):
        result = RunResult(
            status="success",
            question="How many?",
            messages=[{"role": "user", "content": "How many?"}],
            run_metrics=RunMetrics(
                latency_sec=1.0, sql_attempt_count=0, sql_failed_count=0
            ),
        )

        persisted = persisted_result(result)

        self.assertNotIn("messages", persisted)
        self.assertNotIn("question", persisted)
        self.assertEqual(persisted["status"], "success")
        self.assertIn("run_metrics", persisted)

    def test_run_paths_are_timestamped_and_unique(self):
        with TemporaryDirectory() as directory:
            first = next_run_path(Path(directory))
            first.touch()
            second = next_run_path(Path(directory))

        self.assertRegex(first.name, r"^\d{8}T\d{6}\.\d{6}(?:_\d{4})?\.json$")
        self.assertNotEqual(first, second)
        self.assertLess(first.name, second.name)

    def test_run_paths_support_a_prefix(self):
        with TemporaryDirectory() as directory:
            path = next_run_path(Path(directory), "benchmark")

        self.assertRegex(path.name, r"^benchmark_\d{8}T\d{6}\.\d{6}\.json$")


class SqlUsageEvaluatorTests(unittest.TestCase):
    def test_accepts_metadata_answer_without_query(self):
        result = RunResult(
            status="success",
            question="What tables are in this database?",
            answer="races, drivers",
            run_metrics=RunMetrics(
                latency_sec=1.0, sql_attempt_count=0, sql_failed_count=0
            ),
        )
        case = EvalCase(
            name="database_metadata_001",
            question=result.question,
            reference_answer="The database contains races and drivers.",
            sql_required=False,
        )

        evaluation = sql_usage_evaluator(result, case)

        self.assertTrue(evaluation["score"])

    def test_rejects_out_of_scope_answer_that_uses_sql(self):
        result = RunResult(
            status="success",
            question="What is the weather today?",
            answer="I cannot access weather data.",
            sql_attempts=[
                SqlAttempt(query="SELECT 1", result="[(1,)]", succeeded=True)
            ],
            run_metrics=RunMetrics(
                latency_sec=1.0, sql_attempt_count=1, sql_failed_count=0
            ),
        )
        case = EvalCase(
            name="out_of_scope_001",
            question=result.question,
            reference_answer="The agent cannot answer live weather questions.",
            sql_required=False,
        )

        evaluation = sql_usage_evaluator(result, case)

        self.assertFalse(evaluation["score"])


class SqlResultEvaluatorTests(unittest.TestCase):
    @patch("evals.evaluators.db.query", return_value="[(1,), (2,)]")
    def test_ignores_row_order_without_order_by(self, _query):
        result = RunResult(
            status="success",
            question="List the values.",
            answer="1 and 2",
            sql_attempts=[
                SqlAttempt(
                    query="SELECT value FROM values",
                    result="[(2,), (1,)]",
                    succeeded=True,
                )
            ],
            run_metrics=RunMetrics(
                latency_sec=1.0, sql_attempt_count=1, sql_failed_count=0
            ),
        )
        case = EvalCase(
            name="values",
            question=result.question,
            reference_answer="1 and 2",
            gold_sql="SELECT value FROM values",
        )

        evaluation = sql_result_evaluator(result, case)

        self.assertTrue(evaluation["score"])

    @patch("evals.evaluators.db.query", return_value="[(1,), (2,)]")
    def test_preserves_row_order_with_order_by(self, _query):
        result = RunResult(
            status="success",
            question="List the values in order.",
            answer="2 and 1",
            sql_attempts=[
                SqlAttempt(
                    query="SELECT value FROM values ORDER BY value DESC",
                    result="[(2,), (1,)]",
                    succeeded=True,
                )
            ],
            run_metrics=RunMetrics(
                latency_sec=1.0, sql_attempt_count=1, sql_failed_count=0
            ),
        )
        case = EvalCase(
            name="ordered_values",
            question=result.question,
            reference_answer="1 and 2",
            gold_sql="SELECT value FROM values ORDER BY value",
        )

        evaluation = sql_result_evaluator(result, case)

        self.assertFalse(evaluation["score"])


class SqlFailureEvaluatorTests(unittest.TestCase):
    def test_accepts_a_successful_query_after_a_failed_attempt(self):
        result = RunResult(
            status="success",
            question="How many races are there?",
            answer="997",
            sql_attempts=[
                SqlAttempt(query="bad query", result="Error: syntax", succeeded=False),
                SqlAttempt(query="SELECT COUNT(*) FROM races", result="[(997,)]", succeeded=True),
            ],
            run_metrics=RunMetrics(
                latency_sec=1.0, sql_attempt_count=2, sql_failed_count=1
            ),
        )
        case = EvalCase(
            name="race_count",
            question=result.question,
            reference_answer="There are 997 races.",
            max_sql_failures=1,
        )

        self.assertTrue(sql_validity_evaluator(result)["score"])
        self.assertTrue(sql_failure_evaluator(result, case)["score"])

    def test_rejects_too_many_sql_failures(self):
        result = RunResult(
            status="success",
            question="How many races are there?",
            answer="997",
            run_metrics=RunMetrics(
                latency_sec=1.0, sql_attempt_count=3, sql_failed_count=2
            ),
        )
        case = EvalCase(
            name="race_count",
            question=result.question,
            reference_answer="There are 997 races.",
            max_sql_failures=1,
        )

        self.assertFalse(sql_failure_evaluator(result, case)["score"])


if __name__ == "__main__":
    unittest.main()
