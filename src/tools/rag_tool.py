
import json

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from src.tools.generate_answer import AnswerGenerator


class SupportQueryInput(BaseModel):
    question: str = Field(
        ...,
        description="The customer's e-commerce support question.",
        min_length=1,
    )


# Reuse the working custom-data answer generator.
answer_generator = AnswerGenerator()


def search_support_information(question: str) -> str:
    """Retrieve relevant support information and return its answer and source."""
    result = answer_generator.answer(question)

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )


rag_tool = StructuredTool.from_function(
    func=search_support_information,
    name="search_support_information",
    description=(
        "Search the custom e-commerce customer-support dataset "
        "and return the relevant answer, intent, source, and similarity. "
        "Use this tool when a customer asks about orders, delivery, "
        "payments, refunds, returns, accounts, or customer support."
    ),
    args_schema=SupportQueryInput,
)


if __name__ == "__main__":
    test_questions = [
        "How can I cancel my order?",
        "My order has not arrived yet.",
        "What should I do if my payment failed?",
    ]

    for question in test_questions:
        print("\n" + "=" * 65)
        print("QUESTION:", question)
        result = rag_tool.invoke({"question": question})
        print("TOOL RESULT:")
        print(result)
