
def validate_response(result: dict) -> dict:
    """Check whether a support response has usable evidence."""

    answer = str(result.get("answer", "")).strip()
    sources = result.get("sources", [])
    confidence = result.get("confidence", "low")

    issues = []

    if not answer:
        issues.append("Answer is empty.")

    if not sources:
        issues.append("No supporting source was retrieved.")

    similarity = 0.0
    if sources:
        try:
            similarity = float(sources[0].get("similarity", 0.0))
        except (TypeError, ValueError):
            similarity = 0.0

    # A keyword-based forced match can have zero TF-IDF similarity.
    # Check source availability separately from similarity.
    is_valid = bool(answer and sources)

    return {
        "is_valid": is_valid,
        "answer": answer,
        "sources": sources,
        "similarity": similarity,
        "confidence": confidence,
        "issues": issues,
        "needs_escalation": not is_valid,
    }


if __name__ == "__main__":
    examples = [
        {
            "answer": "Check your order status and tracking information.",
            "sources": [
                {
                    "intent": "delayed_delivery",
                    "similarity": 0.0,
                }
            ],
            "confidence": "unvalidated",
        },
        {
            "answer": "I couldn't find a relevant answer.",
            "sources": [],
            "confidence": "low",
        },
    ]

    for example in examples:
        print("\nINPUT:")
        print(example)
        print("VALIDATION:")
        print(validate_response(example))
