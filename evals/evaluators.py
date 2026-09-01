"""Small, deterministic evaluators used by the agent eval suite."""

import json
from typing import Any

from agentevals.trajectory.match import create_trajectory_match_evaluator
from openevals.types import EvaluatorResult


def _as_dict(outputs: Any) -> dict[str, Any]:
    if hasattr(outputs, "model_dump"):
        return outputs.model_dump()
    return outputs


def outcome_evaluator(
    *, outputs: Any, reference_outputs: dict[str, Any], **kwargs: Any
) -> EvaluatorResult:
    """Pass when the answer contains every case-defined expected value."""
    result = _as_dict(outputs)
    answer = str(result.get("answer") or "").casefold()
    expected = reference_outputs.get("expected_answer_contains", [])
    missing = [value for value in expected if str(value).casefold() not in answer]
    score = not missing and result.get("status") == "success"
    comment = (
        None if score else f"Missing expected values: {', '.join(map(str, missing))}"
    )

    return {"key": "outcome", "score": score, "comment": comment}


def sql_validity_evaluator(*, outputs: Any, **kwargs: Any) -> EvaluatorResult:
    """Pass when the agent made a SQL attempt and every attempt executed cleanly."""
    result = _as_dict(outputs)
    attempts = result.get("sql_attempts", [])
    invalid = [attempt for attempt in attempts if not attempt.get("succeeded", False)]
    score = bool(attempts) and not invalid
    comment = None if score else "At least one SQL attempt failed or no SQL was run."

    return {"key": "sql_validity", "score": score, "comment": comment}


def run_metrics_evaluator(
    *, outputs: Any, reference_outputs: dict[str, Any], **kwargs: Any
) -> EvaluatorResult:
    """Pass when the run stays within the case's retry budget."""
    result = _as_dict(outputs)
    metrics = result.get("run_metrics", {})
    retry_count = metrics.get("retry_count", 0)
    max_retries = reference_outputs.get("max_retries", 2)
    score = retry_count <= max_retries
    comment = None if score else f"retry_count={retry_count} exceeds {max_retries}"

    return {"key": "run_metrics", "score": score, "comment": comment}


def trajectory_evaluator(
    *, outputs: Any, reference_outputs: dict[str, Any], **kwargs: Any
) -> EvaluatorResult:
    """Pass when the trajectory contains each tool required by a case."""
    reference = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps({}),
                    }
                }
            ],
        }
        for tool_name in reference_outputs.get("expected_tools", [])
    ]
    evaluator = create_trajectory_match_evaluator(
        trajectory_match_mode="superset", tool_args_match_mode="ignore"
    )
    return evaluator(outputs=outputs, reference_outputs=reference, **kwargs)
