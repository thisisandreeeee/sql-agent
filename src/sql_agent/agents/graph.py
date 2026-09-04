from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .. import tools
from . import graph_nodes as nodes


def build_graph(model):
    graph = StateGraph(MessagesState)

    graph.add_node("list_tables", nodes.list_tables)
    graph.add_node("call_get_schema", nodes.make_get_schema_node(model))
    graph.add_node("get_schema", ToolNode([tools.sql_db_schema], name="get_schema"))
    graph.add_node("generate_query", nodes.make_generate_query_node(model))
    graph.add_node("check_query", nodes.make_check_query_node(model))
    graph.add_node("run_query", ToolNode([tools.sql_db_query], name="run_query"))

    graph.add_edge(START, "list_tables")
    graph.add_edge("list_tables", "call_get_schema")
    graph.add_edge("call_get_schema", "get_schema")
    graph.add_edge("get_schema", "generate_query")
    graph.add_conditional_edges("generate_query", nodes.should_continue)
    graph.add_edge("check_query", "run_query")
    graph.add_edge("run_query", "generate_query")

    return graph.compile()
