
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph.support_graph import support_graph


app = FastAPI(
    title="E-commerce Customer Support API",
    description="API for the multi-agent customer-support workflow.",
    version="1.0.0",
)


class SupportRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The customer's support question.",
    )


class SupportResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    confidence: str = "low"
    recommended_action: str = "automated_support"
    needs_escalation: bool = False


@app.get("/")
def home():
    return {
        "message": "E-commerce Customer Support API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/chat", response_model=SupportResponse)
def chat(request: SupportRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = support_graph.invoke({"question": question})
        final_response = result["final_response"]

        return {
            "question": final_response.get("question", question),
            "answer": final_response.get(
                "answer",
                "Sorry, I could not generate a response.",
            ),
            "sources": final_response.get("sources", []),
            "confidence": final_response.get("confidence", "low"),
            "recommended_action": final_response.get(
                "recommended_action",
                "automated_support",
            ),
            "needs_escalation": final_response.get(
                "needs_escalation",
                False,
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not process support question: {exc}",
        ) from exc
