
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_PATH = PROJECT_ROOT / "artifacts" / "feedback.jsonl"


def save_feedback(
    question: str,
    answer: str,
    rating: int,
    comment: str = "",
    session_id: str = "default",
) -> dict[str, Any]:
    """Validate and save customer feedback as a JSONL record."""

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not answer.strip():
        raise ValueError("Answer cannot be empty.")

    if not 1 <= rating <= 5:
        raise ValueError("Rating must be between 1 and 5.")

    if not session_id.strip():
        raise ValueError("Session ID cannot be empty.")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "question": question.strip(),
        "answer": answer.strip(),
        "rating": rating,
        "comment": comment.strip(),
    }

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    with FEEDBACK_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def get_feedback() -> list[dict[str, Any]]:
    """Read previously saved feedback records."""

    if not FEEDBACK_PATH.exists():
        return []

    records = []

    with FEEDBACK_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


if __name__ == "__main__":
    saved = save_feedback(
        question="Where is my order?",
        answer="Please check your order tracking details.",
        rating=5,
        comment="The response was helpful.",
        session_id="test_session_001",
    )

    print("Feedback saved successfully:")
    print(json.dumps(saved, indent=2, ensure_ascii=False))

    print("\nTotal feedback records:", len(get_feedback()))
    print("Feedback file:", FEEDBACK_PATH)
