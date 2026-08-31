"""Structured evaluation results for the SQL agent."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from . import tools


class SqlAttempt(BaseModel):
    query: str
    result: str | None = None
    succeeded: bool = False


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class EvaluationMetrics(BaseModel):
    latency_sec: float
    sql_attempt_count: int
    retry_count: int


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
    metrics: EvaluationMetrics
    error: EvaluationError | None = None


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def structured_result(
    question: str, state: dict, latency_sec: float
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

    return EvaluationResult(
        status="success" if answer else "incomplete",
        question=question,
        answer=answer,
        sql_attempts=sql_attempts,
        tool_trace=tool_trace,
        metrics=EvaluationMetrics(
            latency_sec=round(latency_sec, 3),
            sql_attempt_count=len(sql_attempts),
            retry_count=max(len(sql_attempts) - 1, 0),
        ),
    )


def error_result(
    question: str, error: Exception, latency_sec: float
) -> EvaluationResult:
    """Create a structured result for a failed agent run."""
    return EvaluationResult(
        status="error",
        question=question,
        metrics=EvaluationMetrics(
            latency_sec=round(latency_sec, 3),
            sql_attempt_count=0,
            retry_count=0,
        ),
        error=EvaluationError(type=type(error).__name__, message=str(error)),
    )
