"""Structured run results for the SQL agent."""

import time
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from . import tools


class SqlAttempt(BaseModel):
    query: str
    result: str | None = None
    succeeded: bool = False


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class RunMetrics(BaseModel):
    latency_sec: float
    sql_attempt_count: int
    retry_count: int
    model_time_sec: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    output_tokens_per_sec: float | None = None


class EvaluationError(BaseModel):
    type: str
    message: str


class EvaluationResult(BaseModel):
    schema_version: str = "1"
    status: Literal["success", "incomplete", "error"]
    question: str
    answer: str | None = None
    sql_attempts: list[SqlAttempt] = Field(default_factory=list)
    tool_trace: list[ToolCall] = Field(default_factory=list)
    run_metrics: RunMetrics
    error: EvaluationError | None = None


class ModelTimingCallback(BaseCallbackHandler):
    """Accumulate time spent in chat-model calls during one agent run."""

    def __init__(self) -> None:
        self._starts: dict[Any, float] = {}
        self._elapsed_sec = 0.0

    @property
    def elapsed_sec(self) -> float:
        return self._elapsed_sec

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs) -> None:
        self._starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        self._finish(run_id)

    def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        self._finish(run_id)

    def _finish(self, run_id) -> None:
        started_at = self._starts.pop(run_id, None)
        if started_at is not None:
            self._elapsed_sec += time.perf_counter() - started_at


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _token_counts(messages) -> dict[str, int | None]:
    usage_records = [
        getattr(message, "usage_metadata", None)
        for message in messages
        if getattr(message, "usage_metadata", None)
    ]
    if not usage_records:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }

    def total(field: str) -> int | None:
        values = [record.get(field) for record in usage_records]
        return sum(values) if all(value is not None for value in values) else None

    input_tokens = total("input_tokens")
    output_tokens = total("output_tokens")
    total_tokens = total("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def structured_result(
    question: str,
    state: dict,
    latency_sec: float,
    model_time_sec: float | None = None,
) -> EvaluationResult:
    """Convert the graph state into a JSON-serializable evaluation result."""
    messages = state.get("messages", [])
    tool_trace: list[ToolCall] = []
    sql_attempts: list[SqlAttempt] = []
    attempts_by_id = {}

    for message in messages:
        for tool_call in getattr(message, "tool_calls", []) or []:
            name = tool_call.get("name", "")
            args = tool_call.get("args", {}) or {}
            tool_trace.append(ToolCall(name=name, args=args))
            if name == tools.sql_db_query.name:
                attempt = SqlAttempt(query=args.get("query", ""))
                sql_attempts.append(attempt)
                if tool_call.get("id"):
                    attempts_by_id[tool_call["id"]] = attempt

        if getattr(message, "type", None) == "tool":
            attempt = attempts_by_id.get(getattr(message, "tool_call_id", None))
            if attempt is not None:
                result = _content_text(getattr(message, "content", ""))
                attempt.result = result
                attempt.succeeded = not result.startswith("Error:")

    answer = None
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and not getattr(
            message, "tool_calls", []
        ):
            answer = _content_text(getattr(message, "content", ""))
            if answer:
                break

    token_counts = _token_counts(messages)
    output_tokens_per_sec = None
    if (
        token_counts["output_tokens"] is not None
        and model_time_sec is not None
        and model_time_sec > 0
    ):
        output_tokens_per_sec = round(
            token_counts["output_tokens"] / model_time_sec, 3
        )

    return EvaluationResult(
        status="success" if answer else "incomplete",
        question=question,
        answer=answer,
        sql_attempts=sql_attempts,
        tool_trace=tool_trace,
        run_metrics=RunMetrics(
            latency_sec=round(latency_sec, 3),
            sql_attempt_count=len(sql_attempts),
            retry_count=max(len(sql_attempts) - 1, 0),
            model_time_sec=(
                round(model_time_sec, 3) if model_time_sec is not None else None
            ),
            **token_counts,
            output_tokens_per_sec=output_tokens_per_sec,
        ),
    )


def error_result(
    question: str,
    error: Exception,
    latency_sec: float,
    model_time_sec: float | None = None,
) -> EvaluationResult:
    """Create a structured result for a failed agent run."""
    return EvaluationResult(
        status="error",
        question=question,
        run_metrics=RunMetrics(
            latency_sec=round(latency_sec, 3),
            sql_attempt_count=0,
            retry_count=0,
            model_time_sec=(
                round(model_time_sec, 3) if model_time_sec is not None else None
            ),
        ),
        error=EvaluationError(type=type(error).__name__, message=str(error)),
    )
