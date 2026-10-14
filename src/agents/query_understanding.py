
import re


INTENT_KEYWORDS = {
    "cancel_order": [
        "cancel order",
        "cancel my order",
        "cancel an order",
    ],
    "track_order": [
        "track order",
        "track my order",
        "tracking status",
        "where is my order",
    ],
    "delayed_delivery": [
        "not arrived",
        "hasn't arrived",
        "has not arrived",
        "not delivered",
        "delivery is late",
        "package is late",
        "where is my package",
    ],
    "get_refund": [
        "refund",
        "money back",
    ],
    "payment_issue": [
        "payment failed",
        "money was deducted",
        "charged twice",
        "payment issue",
    ],
    "recover_password": [
        "reset my password",
        "forgot password",
        "forgot my password",
        "reset password",
    ],
    "contact_human_agent": [
        "human support",
        "human agent",
        "speak to a human",
        "contact customer support",
        "contact support",
    ],
    "return_order": [
        "return a product",
        "return my order",
        "return an item",
    ],
    "damaged_order": [
        "damaged product",
        "arrived damaged",
        "damaged package",
    ],
    "download_invoice": [
        "download invoice",
        "download my bill",
        "get invoice",
    ],
}


CATEGORY_BY_INTENT = {
    "cancel_order": "ORDER",
    "track_order": "ORDER",
    "delayed_delivery": "DELIVERY",
    "get_refund": "REFUND",
    "payment_issue": "PAYMENT",
    "recover_password": "ACCOUNT",
    "contact_human_agent": "CONTACT",
    "return_order": "RETURN",
    "damaged_order": "RETURN",
    "download_invoice": "ORDER",
}


def understand_query(question: str) -> dict:
    """Classify a support question using simple keyword rules."""
    normalized_question = re.sub(
        r"\s+", " ", question.lower()
    ).strip()

    if not normalized_question:
        return {
            "question": question,
            "category": "UNKNOWN",
            "intent": "unknown",
            "needs_escalation": False,
        }

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(
            keyword in normalized_question
            for keyword in keywords
        ):
            return {
                "question": question,
                "category": CATEGORY_BY_INTENT[intent],
                "intent": intent,
                "needs_escalation": False,
            }

    return {
        "question": question,
        "category": "UNKNOWN",
        "intent": "unknown",
        "needs_escalation": True,
    }


if __name__ == "__main__":
    test_questions = [
        "How can I cancel my order?",
        "My order has not arrived yet.",
        "My payment failed but money was deducted.",
        "How do I reset my password?",
        "How can I speak to a human support agent?",
        "What is the weather today?",
    ]

    for question in test_questions:
        print("\nQUESTION:", question)
        print(understand_query(question))
