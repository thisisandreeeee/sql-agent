"""Evaluators used by the agent eval suite."""

import os
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
from evals.types import EvalCase

MODEL = "deepseek-v4-pro"


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
        prompt=CORRECTNESS_PROMPT, feedback_key="correctness", judge=build_model()
    )
    return evaluator(
        inputs=result.question,
        outputs=result.answer,
        reference_outputs=case.reference_answer,
    )


def relevance_evaluator(result: RunResult) -> EvaluatorResult:
    evaluator = create_llm_as_judge(
        prompt=ANSWER_RELEVANCE_PROMPT, feedback_key="relevance", judge=build_model()
    )
    return evaluator(inputs=result.question, outputs=result.answer)


def groundedness_evaluator(result: RunResult) -> EvaluatorResult:
    evaluator = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT, feedback_key="groundedness", judge=build_model()
    )
    latest_attempt = result.sql_attempts[-1]
    context = f"Question: {result.question}\nSQL: {latest_attempt.query}\nResult: {latest_attempt.result}"
    return evaluator(context=context, outputs=result.answer)


def sql_validity_evaluator(result: RunResult) -> EvaluatorResult:
    invalid = [attempt for attempt in result.sql_attempts if not attempt.succeeded]
    score = bool(result.sql_attempts) and not invalid
    comment = None if score else f"Failed SQL queries: {invalid}"
    return EvaluatorResult(key="sql_validity", score=score, comment=comment)


def trajectory_evaluator(result: RunResult) -> EvaluatorResult:
    evaluator = create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT,
        feedback_key="trajectory",
        judge=build_model(),
    )
    return evaluator(outputs=result.messages)
