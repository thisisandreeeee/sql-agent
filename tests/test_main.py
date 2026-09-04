import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from langgraph.graph import END

from sql_agent.agents.graph import MAX_SQL_ATTEMPTS, make_finalize_node, should_continue
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

    def test_treats_middleware_blocked_sql_calls_as_neutral(self):
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
                    content="Tool call limit exceeded. Do not call 'sql_db_query' again.",
                    status="error",
                ),
                SimpleNamespace(type="ai", tool_calls=[], content="I could not finish."),
            ]
        }

        result = structured_result("How many?", state, 1.0)

        self.assertIsNone(result.sql_attempts[0].succeeded)
        self.assertFalse(result.sql_attempts[0].executed)
        self.assertTrue(result.sql_attempts[0].blocked)
        self.assertEqual(result.run_metrics.sql_failed_count, 0)

    def test_routes_missing_query_to_tool_validation(self):
        state = {
            "messages": [
                SimpleNamespace(
                    tool_calls=[{"name": "sql_db_query", "args": {}, "id": "bad"}]
                )
            ]
        }

        self.assertEqual(should_continue(state), "run_query")

    def test_stops_graph_after_bounded_sql_attempts(self):
        tool_calls = [
            {
                "name": "sql_db_query",
                "args": {"query": "SELECT 1"},
                "id": f"query-{index}",
            }
            for index in range(MAX_SQL_ATTEMPTS + 1)
        ]
        state = {
            "messages": [
                SimpleNamespace(tool_calls=tool_calls),
            ]
        }

        self.assertEqual(should_continue(state), "finalize")

    def test_finalize_pairs_pending_tool_calls_and_returns_answer(self):
        model = MagicMock()
        model.invoke.return_value = SimpleNamespace(type="ai", tool_calls=[], content="Done.")
        node = make_finalize_node(model)
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
                )
            ]
        }

        update = node(state)

        self.assertEqual(update["messages"][0].tool_call_id, "query-1")
        self.assertNotEqual(update["messages"][0].status, "error")
        self.assertIn("completed tool results", update["messages"][0].content)
        self.assertEqual(update["messages"][1].content, "Done.")

        result = structured_result(
            "How many?", {"messages": state["messages"] + update["messages"]}, 1.0
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.run_metrics.sql_failed_count, 0)
        self.assertTrue(result.sql_attempts[0].blocked)
        model.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
