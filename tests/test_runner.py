import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evals.runner import main, next_run_path, persisted_result, summarize, write_run

from evals.evaluators import (
    sql_failure_evaluator,
    sql_usage_evaluator,
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
                    {"key": "correctness", "score": 1.0},
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
                    {"key": "correctness", "score": 0.0},
                    {"key": "relevance", "score": 1.0},
                ],
                "error": None,
            },
        ]

        summary = summarize(cases)

        self.assertEqual(summary["total_cases"], 2)
        self.assertEqual(summary["passed_cases"], 0)
        self.assertEqual(summary["latency_sec"], {"total": 4.0, "mean": 2.0})
        self.assertEqual(summary["output_tokens"], {"total": 30, "mean": 15.0})
        self.assertEqual(summary["output_tokens_per_second"], 6.0)
        self.assertEqual(summary["sql_attempt_count"], {"total": 3, "mean": 1.5})
        self.assertEqual(summary["sql_failed_count"], {"total": 0, "mean": 0.0})
        self.assertEqual(
            summary["score_means"], {"correctness": 0.5, "relevance": 0.75}
        )

    def test_passed_cases_require_exact_one_scores(self):
        summary = summarize(
            [
                {
                    "result": None,
                    "evaluations": [
                        {"key": "correctness", "score": 1.0},
                    ],
                    "error": None,
                },
                {
                    "result": None,
                    "evaluations": [
                        {"key": "correctness", "score": 0.5},
                    ],
                    "error": None,
                },
            ]
        )

        self.assertEqual(summary["passed_cases"], 1)

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
            first = next_run_path(Path(directory), "graph")
            first.touch()
            second = next_run_path(Path(directory), "graph")

        self.assertRegex(first.name, r"^graph_\d{8}T\d{6}\.\d{6}(?:_\d{4})?\.json$")
        self.assertNotEqual(first, second)
        self.assertLess(first.name, second.name)

    def test_run_paths_use_the_agent_type(self):
        with TemporaryDirectory() as directory:
            path = next_run_path(Path(directory), "react")

        self.assertRegex(path.name, r"^react_\d{8}T\d{6}\.\d{6}\.json$")


class RunnerPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.cases = [
            EvalCase(
                name=name,
                question=f"Question {name}",
                reference_answer="Answer",
            )
            for name in ("case_1", "case_2", "case_3")
        ]

    @staticmethod
    def record(case, score=1.0, error=None):
        return {
            "case": case.model_dump(mode="json"),
            "result": None,
            "evaluations": [{"key": "test", "score": score}],
            "error": error,
        }

    def test_main_writes_after_each_case(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            calls = []

            def run_case(case, agent):
                calls.append(len(json.loads(path.read_text())["cases"]))
                return self.record(case), [{"key": "test", "score": 1.0}], None

            with (
                patch("evals.runner.load_cases", return_value=self.cases),
                patch("evals.runner.build_model"),
                patch("evals.runner.build_agent", return_value=object()),
                patch("evals.runner._run_case", side_effect=run_case),
            ):
                with patch("evals.runner.next_run_path", return_value=path):
                    self.assertEqual(main(), 0)

            self.assertEqual(calls, [0, 1, 2])
            self.assertEqual(len(json.loads(path.read_text())["cases"]), 3)

    def test_resume_skips_successes_and_retries_failures(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "run.json"
            write_run(
                path,
                "graph",
                None,
                self.cases,
                {
                    "case_1": self.record(self.cases[0]),
                    "case_2": self.record(
                        self.cases[1],
                        error={"type": "RuntimeError", "message": "temporary"},
                    ),
                    "case_3": self.record(self.cases[2], score=0.0),
                },
            )
            calls = []

            def run_case(case, agent):
                calls.append(case.name)
                return self.record(case), [{"key": "test", "score": 1.0}], None

            with (
                patch("evals.runner.load_cases", return_value=self.cases),
                patch("evals.runner.build_model"),
                patch("evals.runner.build_agent", return_value=object()),
                patch("evals.runner._run_case", side_effect=run_case),
            ):
                self.assertEqual(main(resume=path), 0)

            self.assertEqual(calls, ["case_2", "case_3"])
            saved = json.loads(path.read_text())
            self.assertEqual([record["case"]["name"] for record in saved["cases"]], [
                "case_1",
                "case_2",
                "case_3",
            ])


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


class SqlFailureEvaluatorTests(unittest.TestCase):
    def test_accepts_a_successful_query_after_a_failed_attempt(self):
        result = RunResult(
            status="success",
            question="How many races are there?",
            answer="997",
            sql_attempts=[
                SqlAttempt(query="bad query", result="Error: syntax", succeeded=False),
                SqlAttempt(
                    query="SELECT COUNT(*) FROM races",
                    result="[(997,)]",
                    succeeded=True,
                ),
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
