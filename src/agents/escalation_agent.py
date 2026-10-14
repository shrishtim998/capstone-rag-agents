
def decide_escalation(
    intent: str,
    sentiment: str,
    validation: dict,
) -> dict:
    """Decide whether a customer request needs human support."""

    reasons = []

    if validation.get("needs_escalation", False):
        reasons.append("Response validation failed or no source was found.")

    if intent == "unknown":
        reasons.append("The customer request could not be classified.")

    if sentiment == "negative":
        reasons.append("Negative customer sentiment detected.")

    should_escalate = bool(reasons)

    return {
        "needs_escalation": should_escalate,
        "escalation_reasons": reasons,
        "recommended_action": (
            "human_support"
            if should_escalate
            else "automated_support"
        ),
    }


if __name__ == "__main__":
    examples = [
        {
            "intent": "cancel_order",
            "sentiment": "neutral",
            "validation": {
                "is_valid": True,
                "needs_escalation": False,
            },
        },
        {
            "intent": "payment_issue",
            "sentiment": "negative",
            "validation": {
                "is_valid": True,
                "needs_escalation": False,
            },
        },
        {
            "intent": "unknown",
            "sentiment": "neutral",
            "validation": {
                "is_valid": False,
                "needs_escalation": True,
            },
        },
    ]

    for example in examples:
        print("\nINPUT:")
        print(example)
        print("DECISION:")
        print(
            decide_escalation(
                example["intent"],
                example["sentiment"],
                example["validation"],
            )
        )
