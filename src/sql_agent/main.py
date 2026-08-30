"""Run a user query against the SQL agent's chat model."""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from .graph import build_graph

ROOT = Path(__file__).resolve().parents[2]


def build_model():
    """Build the fixed DeepSeek model used by the application and evaluations."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required; copy .env.example to .env")

    return init_chat_model(
        model="deepseek-v4-pro",
        model_provider="deepseek",
        api_key=api_key,
        extra_body={"thinking": {"type": "disabled"}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    args = parser.parse_args()

    model = build_model()
    graph = build_graph(model)
    agent = graph.compile()

    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": args.query}]},
        version="v3",
    )
    for message in stream.messages:
        for token in message.text:
            print(token, end="", flush=True)
    final_state = stream.output
    return final_state


if __name__ == "__main__":
    main()
