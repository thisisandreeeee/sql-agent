import time
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from sql_agent.run_result import ModelTimingCallback, error_result, structured_result
from sql_agent.graph import build_graph
from sql_agent.main import build_model

from evals.evaluators import (
    outcome_evaluator,
    run_metrics_evaluator,
    sql_validity_evaluator,
    trajectory_evaluator,
)


with (Path(__file__).parent / "cases.yaml").open() as cases_file:
    CASES = yaml.safe_load(cases_file)


@pytest.fixture(scope="session")
def model():
    try:
        return build_model()
    except RuntimeError as error:
        pytest.skip(str(error))


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_sql_agent(case, model):
    started_at = time.perf_counter()
    timing = ModelTimingCallback()
    graph = build_graph(model).compile()
    try:
        state = graph.invoke(
            {"messages": [{"role": "user", "content": case["question"]}]},
            config={"callbacks": [timing]},
        )
        result = structured_result(
            case["question"],
            state,
            time.perf_counter() - started_at,
            model_time_sec=timing.elapsed_sec,
        )
    except Exception as error:
        result = error_result(
            case["question"],
            error,
            time.perf_counter() - started_at,
            model_time_sec=timing.elapsed_sec,
        )
        state = {"messages": []}

    outputs = result.model_dump()
    if result.status == "error":
        pytest.fail(result.error.model_dump_json())
    scores = [
        outcome_evaluator(outputs=outputs, reference_outputs=case),
        sql_validity_evaluator(outputs=outputs),
        run_metrics_evaluator(outputs=outputs, reference_outputs=case),
        trajectory_evaluator(
            outputs=state["messages"],
            reference_outputs=case,
        ),
    ]
    assert all(score["score"] for score in scores), {
        "scores": scores,
        "result": outputs,
    }
