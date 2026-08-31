import unittest
from types import SimpleNamespace

from sql_agent.evaluation import structured_result


class StructuredResultTests(unittest.TestCase):
    def test_extracts_answer_sql_result_and_retry_count(self):
        state = {
            "messages": [
                SimpleNamespace(
                    type="ai",
                    tool_calls=[
                        {
                            "name": "sql_db_query",
                            "args": {"query": "SELECT 1"},
                            "id": "query-1",
                        }
                    ],
                ),
                SimpleNamespace(
                    type="tool",
                    tool_call_id="query-1",
                    content="[(1,)]",
                    tool_calls=[],
                ),
                SimpleNamespace(type="ai", tool_calls=[], content="The answer is 1."),
            ]
        }

        result = structured_result("How many?", state, 1.23456)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.answer, "The answer is 1.")
        self.assertEqual(result.sql_attempts[0].query, "SELECT 1")
        self.assertTrue(result.sql_attempts[0].succeeded)
        self.assertEqual(result.metrics.retry_count, 0)
        self.assertEqual(result.metrics.latency_sec, 1.235)
        self.assertEqual(result.model_dump()["schema_version"], "1")


if __name__ == "__main__":
    unittest.main()
