
import re


NEGATIVE_PHRASES = [
    "angry",
    "furious",
    "frustrated",
    "unacceptable",
    "terrible",
    "worst",
    "disappointed",
    "very upset",
    "fed up",
    "still not",
    "failed",
    "deducted",
    "damaged",
    "late",
    "delayed",
    "not arrived",
    "not received",
    "doesn't work",
    "does not work",
    "can't access",
    "cannot access",
    "urgent",
    "complaint",
    "scam",
    "charged twice",
]

POSITIVE_PHRASES = [
    "thank you",
    "thanks",
    "great",
    "excellent",
    "happy",
    "appreciate",
    "resolved",
    "working now",
    "very helpful",
]


def analyze_sentiment(question: str) -> dict:
    """Classify customer sentiment using simple keyword rules."""
    normalized = re.sub(
        r"\s+", " ", question.lower()
    ).strip()

    if not normalized:
        return {
            "question": question,
            "sentiment": "neutral",
            "priority": "normal",
            "reason": "No question was provided.",
        }

    negative_matches = [
        phrase
        for phrase in NEGATIVE_PHRASES
        if phrase in normalized
    ]

    positive_matches = [
        phrase
        for phrase in POSITIVE_PHRASES
        if phrase in normalized
    ]

    if negative_matches:
        sentiment = "negative"
        priority = "high"
        reason = (
            "Negative language detected: "
            + ", ".join(negative_matches)
        )
    elif positive_matches:
        sentiment = "positive"
        priority = "low"
        reason = (
            "Positive language detected: "
            + ", ".join(positive_matches)
        )
    else:
        sentiment = "neutral"
        priority = "normal"
        reason = "No clear positive or negative keywords detected."

    return {
        "question": question,
        "sentiment": sentiment,
        "priority": priority,
        "reason": reason,
    }


if __name__ == "__main__":
    test_questions = [
        "Thank you, my issue is resolved!",
        "My payment failed and I am very frustrated.",
        "My order has not arrived yet.",
        "How can I cancel my order?",
        "This is unacceptable and I am furious!",
    ]

    for question in test_questions:
        print("\nQUESTION:", question)
        print(analyze_sentiment(question))
