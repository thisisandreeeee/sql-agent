from . import graph, react


def build_agent(model, agent_type: str = "graph"):
    if agent_type == "graph":
        return graph.build_graph(model)
    if agent_type == "react":
        return react.build_react_agent(model)
    raise ValueError(
        f"Unknown agent_type {agent_type!r}; expected 'graph' or 'react'"
    )


__all__ = ["build_agent"]
