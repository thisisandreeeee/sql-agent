import unittest
from types import SimpleNamespace

from sql_agent.agents.graph import should_continue
from sql_agent.types import structured_result


class StructuredResultTests(unittest.TestCase):
    def test_extracts_answer_and_sql_metrics(self):
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
                SimpleNamespace(
                    type="ai",
                    tool_calls=[],
                    content="The answer is 1.",
                    usage_metadata={
                        "input_tokens": 10,
                        "output_tokens": 4,
                        "total_tokens": 14,
                    },
                ),
            ]
        }

        result = structured_result("How many?", state, 1.23456, model_time_sec=0.25)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.answer, "The answer is 1.")
        self.assertEqual(result.messages, state["messages"])
        self.assertEqual(result.sql_attempts[0].query, "SELECT 1")
        self.assertTrue(result.sql_attempts[0].succeeded)
        self.assertEqual(result.run_metrics.sql_attempt_count, 1)
        self.assertEqual(result.run_metrics.sql_failed_count, 0)
        self.assertEqual(result.run_metrics.latency_sec, 1.235)
        self.assertEqual(result.run_metrics.input_tokens, 10)
        self.assertEqual(result.run_metrics.output_tokens, 4)
        self.assertEqual(result.run_metrics.total_tokens, 14)
        self.assertEqual(result.run_metrics.output_tokens_per_sec, 16.0)
        self.assertEqual(result.model_dump()["schema_version"], "2")

    def test_counts_failed_sql_attempts_instead_of_retries(self):
        state = {
            "messages": [
                SimpleNamespace(
                    type="ai",
                    tool_calls=[
                        {
                            "name": "sql_db_query",
                            "args": {"query": "bad query"},
                            "id": "query-1",
                        }
                    ],
                ),
                SimpleNamespace(
                    type="tool",
                    tool_call_id="query-1",
                    content="Error: syntax",
                    tool_calls=[],
                ),
                SimpleNamespace(
                    type="ai",
                    tool_calls=[
                        {
                            "name": "sql_db_query",
                            "args": {"query": "SELECT 1"},
                            "id": "query-2",
                        }
                    ],
                ),
                SimpleNamespace(
                    type="tool",
                    tool_call_id="query-2",
                    content="[(1,)]",
                    tool_calls=[],
                ),
                SimpleNamespace(type="ai", tool_calls=[], content="The answer is 1."),
            ]
        }

        result = structured_result("How many?", state, 1.0)

        self.assertEqual(result.run_metrics.sql_attempt_count, 2)
        self.assertEqual(result.run_metrics.sql_failed_count, 1)

    def test_routes_missing_query_to_tool_validation(self):
        state = {
            "messages": [
                SimpleNamespace(
                    tool_calls=[{"name": "sql_db_query", "args": {}, "id": "bad"}]
                )
            ]
        }

        self.assertEqual(should_continue(state), "run_query")


if __name__ == "__main__":
    unittest.main()
