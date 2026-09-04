from typing import Literal
from uuid import uuid4

from langchain.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .. import tools


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
then look at the results of the query and return the answer.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".format(
    dialect="sqlite"
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


def should_continue(state: MessagesState) -> Literal[END, "run_query"]:
    tool_calls = state["messages"][-1].tool_calls
    if not tool_calls:
        return END
    return "run_query"


def build_graph(model):
    graph = StateGraph(MessagesState)

    graph.add_node("list_tables", list_tables)
    graph.add_node("call_get_schema", make_get_schema_node(model))
    graph.add_node("get_schema", ToolNode([tools.sql_db_schema], name="get_schema"))
    graph.add_node("generate_query", make_generate_query_node(model))
    graph.add_node("run_query", ToolNode([tools.sql_db_query], name="run_query"))

    graph.add_edge(START, "list_tables")
    graph.add_edge("list_tables", "call_get_schema")
    graph.add_edge("call_get_schema", "get_schema")
    graph.add_edge("get_schema", "generate_query")
    graph.add_conditional_edges("generate_query", should_continue)
    graph.add_edge("run_query", "generate_query")

    return graph.compile()
