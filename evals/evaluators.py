"""Evaluators used by the agent eval suite."""

import ast
import os
import re
from collections import Counter
from pathlib import Path

from agentevals.trajectory.llm import (
    create_trajectory_llm_as_judge,
    TRAJECTORY_ACCURACY_PROMPT,
)
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from openevals.llm import create_llm_as_judge
from openevals.prompts import (
    CORRECTNESS_PROMPT,
    ANSWER_RELEVANCE_PROMPT,
    RAG_GROUNDEDNESS_PROMPT,
)
from openevals.types import EvaluatorResult

from sql_agent.types import RunResult
from sql_agent import db
from evals.types import EvalCase

MODEL = "deepseek-v4-pro"
JUDGE_SCORE_CHOICES = [0.0, 0.5, 1.0]

CORRECTNESS_RUBRIC = """
Score correctness of the answer against the reference answer:
- 0.0: The answer is incorrect or fails to answer the question.
- 0.5: The answer is partially correct but has a meaningful factual error or omission.
- 1.0: The answer is correct and complete.
Use exactly one of 0.0, 0.5, or 1.0.
"""

RELEVANCE_RUBRIC = """
Score how directly the answer addresses the user's question:
- 0.0: The answer is irrelevant or does not address the question.
- 0.5: The answer addresses the question but includes a meaningful omission or digression.
- 1.0: The answer directly and fully addresses the question.
Use exactly one of 0.0, 0.5, or 1.0.
"""

GROUNDEDNESS_RUBRIC = """
Score whether the answer is supported by the supplied SQL result:
- 0.0: The answer is unsupported or contradicts the SQL result.
- 0.5: The answer is partly supported but includes a meaningful unsupported claim or omission.
- 1.0: The answer is fully supported by the SQL result.
Use exactly one of 0.0, 0.5, or 1.0.
"""

TRAJECTORY_RUBRIC = """
Score the quality of the agent's trajectory:
- 0.0: The trajectory fails to reach the goal or is fundamentally misguided.
- 0.5: The trajectory reaches part of the goal but has a meaningful reasoning or efficiency problem.
- 1.0: The trajectory reaches the goal with sound, reasonably efficient reasoning.
Use exactly one of 0.0, 0.5, or 1.0.
"""


def build_model():
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required; copy .env.example to .env")
    return init_chat_model(
        model=MODEL,
        model_provider="deepseek",
        api_key=api_key,
        extra_body={"thinking": {"type": "disabled"}},
    )


def correctness_evaluator(result: RunResult, case: EvalCase) -> EvaluatorResult:
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT + CORRECTNESS_RUBRIC,
        feedback_key="correctness",
        judge=build_model(),
        choices=JUDGE_SCORE_CHOICES,
    )
    return evaluator(
        inputs=result.question,
        outputs=result.answer,
        reference_outputs=case.reference_answer,
    )


def relevance_evaluator(result: RunResult) -> EvaluatorResult:
    evaluator = create_llm_as_judge(
        prompt=ANSWER_RELEVANCE_PROMPT + RELEVANCE_RUBRIC,
        feedback_key="relevance",
        judge=build_model(),
        choices=JUDGE_SCORE_CHOICES,
    )
    return evaluator(inputs=result.question, outputs=result.answer)


def groundedness_evaluator(result: RunResult) -> EvaluatorResult:
    evaluator = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT + GROUNDEDNESS_RUBRIC,
        feedback_key="groundedness",
        judge=build_model(),
        choices=JUDGE_SCORE_CHOICES,
    )
    latest_attempt = result.sql_attempts[-1]
    context = f"Question: {result.question}\nSQL: {latest_attempt.query}\nResult: {latest_attempt.result}"
    return evaluator(context=context, outputs=result.answer)


def sql_usage_evaluator(result: RunResult, case: EvalCase) -> EvaluatorResult:
    used_sql = bool(result.sql_attempts)
    score = used_sql == case.sql_required
    if score:
        comment = None
    elif case.sql_required:
        comment = "Expected the agent to query the database, but it did not."
    else:
        comment = "Expected the agent to answer without a SQL query."
    return EvaluatorResult(key="sql_usage", score=score, comment=comment)


def sql_result_evaluator(result: RunResult, case: EvalCase) -> EvaluatorResult:
    gold_result = db.query(case.gold_sql or "")
    actual_result = result.sql_attempts[-1].result
    try:
        expected_rows = ast.literal_eval(gold_result)
        actual_rows = ast.literal_eval(actual_result or "")
        ordered = bool(re.search(r"\border\s+by\b", case.gold_sql or "", re.I))
        score = (
            actual_rows == expected_rows
            if ordered
            else Counter(actual_rows) == Counter(expected_rows)
        )
    except (SyntaxError, ValueError, TypeError):
        score = actual_result == gold_result
    comment = None if score else "The query result did not match the gold query."
    return EvaluatorResult(key="sql_result", score=score, comment=comment)


def sql_failure_evaluator(result: RunResult, case: EvalCase) -> EvaluatorResult:
    failures = result.run_metrics.sql_failed_count
    score = failures <= case.max_sql_failures
    comment = (
        None
        if score
        else f"Used {failures} failed SQL attempts; maximum is {case.max_sql_failures}."
    )
    return EvaluatorResult(key="sql_failure_limit", score=score, comment=comment)


def trajectory_evaluator(result: RunResult) -> EvaluatorResult:
    evaluator = create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT + TRAJECTORY_RUBRIC,
        feedback_key="trajectory",
        judge=build_model(),
        choices=JUDGE_SCORE_CHOICES,
    )
    return evaluator(outputs=result.messages)
