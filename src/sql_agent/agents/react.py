from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

from .. import tools


REACT_SYSTEM_PROMPT = """
You are an agent that answers questions using a SQLite database.

You have tools for:
- listing database tables
- inspecting table schemas
- executing SQL queries

Use these tools as needed to answer the user's question.

Before querying unfamiliar tables, inspect their schemas.
Only execute read-only SQL.
Never execute INSERT, UPDATE, DELETE, DROP, ALTER, or other
statements that modify the database.

When a query fails, inspect the error, correct the query, and retry.

If the database does not contain sufficient information to answer the user's question, say so.
Do not guess, infer unsupported facts, or continue querying once you have established that the required data is unavailable.

In the final answer, make only factual claims supported by information returned by the database tools.
Do not present assumptions, prior knowledge, or unsupported inferences as facts.
Base any calculations or aggregations only on retrieved data, and make clear when an answer is derived.
Before answering, check each factual claim against the retrieved data and omit or qualify any claim it does not support.

Once you have enough information, answer the user's question directly.
"""


def build_react_agent(model):
    return create_agent(
        model=model,
        tools=[
            tools.sql_db_list_tables,
            tools.sql_db_schema,
            tools.sql_db_query,
        ],
        system_prompt=REACT_SYSTEM_PROMPT,
        middleware=[ToolCallLimitMiddleware(tool_name="sql_db_query", run_limit=10)],
    )
