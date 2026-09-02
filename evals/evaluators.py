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
from openevals.prompts import CORRECTNESS_PROMPT
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


def trajectory_evaluator(result: RunResult) -> EvaluatorResult:
    evaluator = create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT,
        feedback_key="trajectory",
        judge=build_model(),
    )
    return evaluator(outputs=result.messages)
