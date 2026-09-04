import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import yaml
from openevals.types import EvaluatorResult

from sql_agent.agents import build_agent
from sql_agent.main import build_model
from sql_agent.types import ModelTimingCallback, RunResult, structured_result

from evals.evaluators import (
    correctness_evaluator,
    relevance_evaluator,
    groundedness_evaluator,
    sql_validity_evaluator,
    sql_result_evaluator,
    sql_usage_evaluator,
    sql_failure_evaluator,
    trajectory_evaluator,
)
from evals.types import EvalCase


def main(agent_type: str = "graph") -> int:
    with (Path(__file__).parent / "cases.yaml").open() as cases_file:
        cases = [EvalCase.model_validate(case) for case in yaml.safe_load(cases_file)]

    agent = build_agent(build_model(), agent_type)
    failed = 0
    records = []
    for case in cases:
        record, scores, error = _run_case(case, agent)
        records.append(record)
        if error is not None:
            print(f"FAIL {case.name}: {error}")
            failed += 1
            continue

        passed = all(score["score"] == 1.0 for score in scores)
        print(f"{'PASS' if passed else 'FAIL'} {case.name}")
        for score in scores:
            print(f"  {score['key']}: {score['score']}")
        if not passed:
            print(scores)
            failed += 1

    run = {
        "agent_type": agent_type,
        "summary": summarize(records),
        "cases": records,
    }
    run_path = next_run_path(
        Path(__file__).resolve().parents[1] / "runs", agent_type
    )
    run_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(f"Saved evaluation run to {run_path}")
    return int(failed > 0)


def run_case(case: EvalCase, agent) -> list[EvaluatorResult]:
    """Run one case and return its evaluations."""
    _, evaluations, error = _run_case(case, agent)
    if error is not None:
        raise error
    return evaluations


def persisted_result(result: RunResult | None) -> dict | None:
    if result is None:
        return None
    return result.model_dump(mode="json", exclude={"messages", "question"})


def _run_case(
    case: EvalCase, agent
) -> tuple[dict, list[EvaluatorResult], Exception | None]:
    result = None
    evaluations = []
    error = None
    try:
        started_at = time.perf_counter()
        timing = ModelTimingCallback()
        state = agent.invoke(
            {"messages": [{"role": "user", "content": case.question}]},
            config={"callbacks": [timing], "recursion_limit": 50},
        )
        result = structured_result(
            case.question,
            state,
            time.perf_counter() - started_at,
            model_time_sec=timing.elapsed_sec,
        )
        if case.reference_answer:
            evaluations.append(correctness_evaluator(result, case))
        evaluations.append(relevance_evaluator(result))
        evaluations.append(trajectory_evaluator(result))
        evaluations.append(sql_usage_evaluator(result, case))
        evaluations.append(sql_failure_evaluator(result, case))
        if result.sql_attempts:
            evaluations.append(groundedness_evaluator(result))
            evaluations.append(sql_validity_evaluator(result))
            if case.gold_sql:
                evaluations.append(sql_result_evaluator(result, case))
    except Exception as exc:
        error = exc

    try:
        serialized_result = persisted_result(result)
        serialized_evaluations = [dict(score) for score in evaluations]
    except Exception as exc:
        serialized_result = None
        serialized_evaluations = []
        error = exc

    return (
        {
            "case": case.model_dump(mode="json"),
            "result": serialized_result,
            "evaluations": serialized_evaluations,
            "error": (
                {"type": type(error).__name__, "message": str(error)}
                if error is not None
                else None
            ),
        },
        evaluations,
        error,
    )


def next_run_path(runs_dir: Path, agent_type: str) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S.%f")
    path = runs_dir / f"{agent_type}_{timestamp}.json"
    suffix = 1
    while path.exists():
        path = runs_dir / f"{agent_type}_{timestamp}_{suffix:04d}.json"
        suffix += 1
    return path


def _metric_summary(metrics: list[dict], field: str) -> dict[str, float | int | None]:
    values = [metric[field] for metric in metrics if metric.get(field) is not None]
    return (
        {"total": round(sum(values), 3), "mean": round(sum(values) / len(values), 3)}
        if values
        else {"total": None, "mean": None}
    )


def summarize(cases: list[dict]) -> dict:
    metrics = [
        case["result"]["run_metrics"] for case in cases if case["result"] is not None
    ]
    passed_cases = sum(
        case["error"] is None
        and all(score.get("score") == 1.0 for score in case["evaluations"])
        for case in cases
    )
    summary = {
        "total_cases": len(cases),
        "passed_cases": passed_cases,
        **{
            field: _metric_summary(metrics, field)
            for field in (
                "latency_sec",
                "model_time_sec",
                "output_tokens",
                "input_tokens",
                "total_tokens",
                "sql_attempt_count",
                "sql_failed_count",
            )
        },
    }

    rates = [
        (metric["output_tokens"], metric["model_time_sec"])
        for metric in metrics
        if metric.get("output_tokens") is not None
        and metric.get("model_time_sec") is not None
    ]
    output_tokens = [output for output, _ in rates]
    model_times = [model_time for _, model_time in rates]
    summary["output_tokens_per_second"] = (
        round(sum(output_tokens) / sum(model_times), 3)
        if output_tokens and model_times and sum(model_times) > 0
        else None
    )

    scores: dict[str, list[float]] = {}
    for case in cases:
        for evaluation in case["evaluations"]:
            score = evaluation.get("score")
            if isinstance(score, (int, float)):
                scores.setdefault(evaluation["key"], []).append(float(score))
    summary["score_means"] = {
        key: round(sum(values) / len(values), 3) for key, values in scores.items()
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-type", choices=("graph", "react"), default="graph")
    raise SystemExit(main(parser.parse_args().agent_type))
