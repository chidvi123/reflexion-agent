from typing import List
from pydantic import BaseModel,Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing . ")
    superfluous : str = Field(description="Critique of what is superfluous ")


class AnswerQuestion(BaseModel):
    """Provide an initial answer and critique."""

    answer: str = Field(
        description=(
            "Only the answer to the user's question, approximately "
            "150 words. Do not include reflection or search queries "
            "inside this field."
        )
    )

    reflection: Reflection = Field(
        description=(
            "Critique the answer. Populate only missing and "
            "superfluous; do not include search queries here."
        )
    )

    search_queries: List[str] = Field(
        description=(
            "A top-level list containing 1-3 search queries "
            "to improve the answer."
        )
    )


class ReviseAnswer(AnswerQuestion):
    references : List[str] = Field(
        description="Citations motivating your updated answer . "
    )