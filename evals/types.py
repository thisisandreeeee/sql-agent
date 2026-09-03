from pydantic import BaseModel


class EvalCase(BaseModel):
    name: str
    question: str
    reference_answer: str
    gold_sql: str | None = None
    sql_required: bool = True
    max_retries: int = 2
