import time
from pathlib import Path

import yaml
from openevals.types import EvaluatorResult

from sql_agent.graph import build_graph
from sql_agent.main import build_model
from sql_agent.types import ModelTimingCallback, structured_result

from evals.evaluators import (
    correctness_evaluator,
    relevance_evaluator,
    groundedness_evaluator,
    sql_validity_evaluator,
    trajectory_evaluator,
)
from evals.types import EvalCase


def main() -> int:
    with (Path(__file__).parent / "cases.yaml").open() as cases_file:
        cases = [EvalCase.model_validate(case) for case in yaml.safe_load(cases_file)]

    graph = build_graph(build_model()).compile()
    failed = 0
    for case in cases:
        try:
            scores = run_case(case, graph)
            passed = all(score["score"] for score in scores)
            print(f"{'PASS' if passed else 'FAIL'} {case.name}")
            for score in scores:
                print(f"  {score['key']}: {score['score']}")

            if not passed:
                print(scores)
                failed += 1
        except Exception as error:
            print(f"FAIL {case.name}: {error}")
            failed += 1

    return int(failed > 0)


def run_case(case: EvalCase, graph) -> list[EvaluatorResult]:
    started_at = time.perf_counter()
    timing = ModelTimingCallback()
    state = graph.invoke(
        {"messages": [{"role": "user", "content": case.question}]},
        config={"callbacks": [timing]},
    )
    result = structured_result(
        case.question,
        state,
        time.perf_counter() - started_at,
        model_time_sec=timing.elapsed_sec,
    )
    evaluations = [
        correctness_evaluator(result, case),
        relevance_evaluator(result),
        trajectory_evaluator(result),
    ]
    if result.sql_attempts:
        evaluations.extend(
            [groundedness_evaluator(result), sql_validity_evaluator(result)]
        )

    return evaluations


if __name__ == "__main__":
    raise SystemExit(main())
