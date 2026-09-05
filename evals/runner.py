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
    sql_usage_evaluator,
    sql_failure_evaluator,
    trajectory_evaluator,
)
from evals.types import EvalCase

CASE_GROUPS = ("basic", "advanced", "behavioral")


def load_cases(path: Path | None = None, group: str | None = None) -> list[EvalCase]:
    """Load grouped cases while keeping EvalCase's public schema unchanged."""
    cases_path = path or Path(__file__).parent / "cases.yaml"
    with cases_path.open(encoding="utf-8") as cases_file:
        grouped_cases = yaml.safe_load(cases_file)

    if not isinstance(grouped_cases, dict):
        raise ValueError("cases.yaml must contain a mapping of case groups")

    unknown_groups = set(grouped_cases) - set(CASE_GROUPS)
    if unknown_groups:
        raise ValueError(f"Unknown case group(s): {sorted(unknown_groups)}")

    selected_groups = CASE_GROUPS if group is None else (group,)
    if group is not None and group not in CASE_GROUPS:
        raise ValueError(f"Unknown case group: {group}")

    cases = []
    for case_group in selected_groups:
        entries = grouped_cases.get(case_group)
        if not isinstance(entries, list):
            raise ValueError(f"Case group {case_group!r} must be a list")
        cases.extend(EvalCase.model_validate(entry) for entry in entries)

    names = [case.name for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case name(s): {duplicates}")
    return cases


def main(
    agent_type: str = "graph",
    group: str | None = None,
    resume: Path | None = None,
) -> int:
    cases = load_cases(group=group)

    run_path = Path(
        resume
        or next_run_path(Path(__file__).resolve().parents[1] / "runs", agent_type)
    )
    if resume is not None:
        loaded_run = load_run(run_path, agent_type, cases, group)
        records_by_name = {
            record["case"]["name"]: record for record in loaded_run["cases"]
        }
    else:
        if run_path.exists():
            raise FileExistsError(
                f"Evaluation run already exists: {run_path}; use --resume to continue it"
            )
        records_by_name = {}

    write_run(run_path, agent_type, group, cases, records_by_name)

    pending = [
        case
        for case in cases
        if case.name not in records_by_name
        or _record_failed(records_by_name[case.name])
    ]
    agent = build_agent(build_model(), agent_type) if pending else None
    for case in cases:
        existing = records_by_name.get(case.name)
        if existing is not None and not _record_failed(existing):
            print(f"SKIP {case.name} (already complete)")
            continue

        record, scores, error = _run_case(case, agent)
        records_by_name[case.name] = record
        write_run(run_path, agent_type, group, cases, records_by_name)
        _print_case_status(case, scores, error)

    records = _ordered_records(cases, records_by_name)
    failed = sum(_record_failed(record) for record in records)
    print(f"Saved evaluation run to {run_path}")
    return int(failed > 0)


def _print_case_status(
    case: EvalCase, scores: list[EvaluatorResult], error: Exception | None
) -> None:
    if error is not None:
        print(f"FAIL {case.name}: {error}")
        return

    passed = all(score["score"] == 1.0 for score in scores)
    print(f"{'PASS' if passed else 'FAIL'} {case.name}")
    for score in scores:
        print(f"  {score['key']}: {score['score']}")
    if not passed:
        print(scores)


def _record_failed(record: dict) -> bool:
    return record.get("error") is not None or not all(
        score.get("score") == 1.0 for score in record.get("evaluations", [])
    )


def _ordered_records(
    cases: list[EvalCase], records_by_name: dict[str, dict]
) -> list[dict]:
    return [
        records_by_name[case.name]
        for case in cases
        if case.name in records_by_name
    ]


def write_run(
    path: Path,
    agent_type: str,
    group: str | None,
    cases: list[EvalCase],
    records_by_name: dict[str, dict],
) -> None:
    """Persist the current run atomically so it is valid after every case."""
    records = _ordered_records(cases, records_by_name)
    run = {
        "agent_type": agent_type,
        "group": group,
        "summary": summarize(records),
        "cases": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def load_run(
    path: Path, agent_type: str, cases: list[EvalCase], group: str | None
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation run does not exist: {path}")

    run = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(run, dict) or not isinstance(run.get("cases"), list):
        raise ValueError(f"Invalid evaluation run: {path}")
    if run.get("agent_type") != agent_type:
        raise ValueError(
            f"Run uses agent type {run.get('agent_type')!r}, expected {agent_type!r}"
        )
    if "group" in run and run["group"] != group:
        raise ValueError(f"Run uses group {run['group']!r}, expected {group!r}")

    expected = {case.name: case for case in cases}
    seen = set()
    for record in run["cases"]:
        if not isinstance(record, dict) or not isinstance(record.get("case"), dict):
            raise ValueError(f"Invalid case record in evaluation run: {path}")
        name = record["case"].get("name")
        if name not in expected:
            raise ValueError(f"Run contains case outside the selected suite: {name!r}")
        if name in seen:
            raise ValueError(f"Run contains duplicate case: {name!r}")
        if EvalCase.model_validate(record["case"]) != expected[name]:
            raise ValueError(
                f"Case definition changed since the run was created: {name!r}"
            )
        seen.add(name)
    return run


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
    parser.add_argument("--group", choices=CASE_GROUPS)
    parser.add_argument(
        "--resume", type=Path, help="Resume an existing run at this path"
    )
    args = parser.parse_args()
    raise SystemExit(main(args.agent_type, args.group, resume=args.resume))
