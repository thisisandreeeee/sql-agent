from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from . import nodes, tools


def build_graph(model):
    g = StateGraph(MessagesState)

    g.add_node("list_tables", nodes.list_tables)
    g.add_node("call_get_schema", nodes.make_get_schema_node(model))
    g.add_node("get_schema", ToolNode([tools.sql_db_schema], name="get_schema"))
    g.add_node("generate_query", nodes.make_generate_query_node(model))
    g.add_node("check_query", nodes.make_check_query_node(model))
    g.add_node("run_query", ToolNode([tools.sql_db_query], name="run_query"))

    g.add_edge(START, "list_tables")
    g.add_edge("list_tables", "call_get_schema")
    g.add_edge("call_get_schema", "get_schema")
    g.add_edge("get_schema", "generate_query")
    g.add_conditional_edges("generate_query", nodes.should_continue)
    g.add_edge("check_query", "run_query")
    g.add_edge("run_query", "generate_query")

    return g
