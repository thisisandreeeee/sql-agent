from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    name: str
    question: str
    reference_answer: str
    expected_tools: list[str] = Field(default_factory=list)
    max_retries: int = 2
