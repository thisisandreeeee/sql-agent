"""Run a user query against the SQL agent's chat model."""

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from .evaluation import ModelTimingCallback, error_result, structured_result
from .graph import build_graph

ROOT = Path(__file__).resolve().parents[2]


def build_model():
    """Build the fixed DeepSeek model used by the application and evaluations."""
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required; copy .env.example to .env")

    return init_chat_model(
        model="deepseek-v4-flash",
        model_provider="deepseek",
        api_key=api_key,
        extra_body={"thinking": {"type": "disabled"}},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the structured evaluation result to this JSON file.",
    )
    args = parser.parse_args()

    started_at = time.perf_counter()
    model_timing = ModelTimingCallback()
    try:
        model = build_model()
        graph = build_graph(model)
        final_state = graph.compile().invoke(
            {"messages": [{"role": "user", "content": args.query}]},
            config={"callbacks": [model_timing]},
        )
        result = structured_result(
            args.query,
            final_state,
            time.perf_counter() - started_at,
            model_time_sec=model_timing.elapsed_sec,
        )
        print(result.answer or "")
        exit_code = 0
    except Exception as error:
        result = error_result(
            args.query,
            error,
            time.perf_counter() - started_at,
            model_time_sec=model_timing.elapsed_sec,
        )
        print(f"Error: {error}")
        exit_code = 1

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
