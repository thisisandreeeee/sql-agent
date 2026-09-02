from typing import Literal
from uuid import uuid4

from langchain.messages import AIMessage
from langgraph.graph import END, MessagesState

from . import tools


def list_tables(state: MessagesState):
    tool_call = {
        "name": tools.sql_db_list_tables.name,
        "args": {},
        "id": f"call_{uuid4().hex}",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    tool_message = tools.sql_db_list_tables.invoke(tool_call)
    return {"messages": [tool_call_message, tool_message]}


def make_get_schema_node(model):
    def get_schema(state: MessagesState):
        llm_with_tools = model.bind_tools(
            [tools.sql_db_schema], tool_choice="any", strict=True
        )
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    return get_schema


GENERATE_QUERY_SYSTEM_PROMPT = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".format(
    dialect="sqlite", top_k=5
)


def make_generate_query_node(model):
    def generate_query(state: MessagesState):
        system_message = {
            "role": "system",
            "content": GENERATE_QUERY_SYSTEM_PROMPT,
        }
        llm_with_tools = model.bind_tools([tools.sql_db_query], strict=True)
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    return generate_query


CHECK_QUERY_SYSTEM_PROMPT = """
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.
You will call the appropriate tool to execute the query after running this check.
""".format(
    dialect="sqlite"
)


def make_check_query_node(model):
    def check_query(state: MessagesState):
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = model.bind_tools(
            [tools.sql_db_query], tool_choice="any", strict=True
        )
        response = llm_with_tools.invoke(
            [{"role": "system", "content": CHECK_QUERY_SYSTEM_PROMPT}, user_message]
        )
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    return check_query


def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    return "check_query" if state["messages"][-1].tool_calls else END
