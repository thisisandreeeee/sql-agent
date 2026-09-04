import unittest
from unittest.mock import MagicMock, patch

from sql_agent import tools
from sql_agent.agents import build_agent
from sql_agent.agents.react import REACT_SYSTEM_PROMPT, build_react_agent


class AgentFactoryTests(unittest.TestCase):
    @patch("sql_agent.agents.graph.build_graph")
    def test_builds_graph_agent(self, build_graph):
        expected = object()
        build_graph.return_value = expected

        result = build_agent(MagicMock(), "graph")

        self.assertIs(result, expected)
        build_graph.assert_called_once()

    @patch("sql_agent.agents.react.build_react_agent")
    def test_builds_react_agent(self, build_react):
        expected = object()
        build_react.return_value = expected

        result = build_agent(MagicMock(), "react")

        self.assertIs(result, expected)
        build_react.assert_called_once()

    def test_rejects_unknown_agent_type(self):
        with self.assertRaises(ValueError):
            build_agent(MagicMock(), "unknown")


class ReactAgentTests(unittest.TestCase):
    @patch("sql_agent.agents.react.create_agent")
    def test_uses_sql_inspection_and_query_tools(self, create_agent):
        model = MagicMock()
        build_react_agent(model)

        create_agent.assert_called_once_with(
            model=model,
            tools=[
                tools.sql_db_list_tables,
                tools.sql_db_schema,
                tools.sql_db_query,
            ],
            system_prompt=REACT_SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
