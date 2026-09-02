import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evals.runner import next_run_path, persisted_result, summarize
from sql_agent.types import RunMetrics, RunResult


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
                        "retry_count": 0,
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
                        "retry_count": 1,
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
        self.assertEqual(summary["retry_count"], 1)
        self.assertEqual(
            summary["score_means"], {"correctness": 0.5, "relevance": 0.75}
        )

    def test_persisted_result_excludes_messages_and_duplicate_question(self):
        result = RunResult(
            status="success",
            question="How many?",
            messages=[{"role": "user", "content": "How many?"}],
            run_metrics=RunMetrics(
                latency_sec=1.0, sql_attempt_count=0, retry_count=0
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


if __name__ == "__main__":
    unittest.main()
