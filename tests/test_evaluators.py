from evals.evaluators import (
    outcome_evaluator,
    run_metrics_evaluator,
    sql_validity_evaluator,
)


def test_deterministic_evaluators_score_a_successful_run():
    result = {
        "status": "success",
        "answer": "Ferrari won 230 races.",
        "sql_attempts": [{"succeeded": True}],
        "run_metrics": {"retry_count": 0},
    }
    reference = {"expected_answer_contains": ["Ferrari", "230"], "max_retries": 1}

    assert outcome_evaluator(outputs=result, reference_outputs=reference)["score"]
    assert sql_validity_evaluator(outputs=result)["score"]
    assert run_metrics_evaluator(outputs=result, reference_outputs=reference)["score"]
