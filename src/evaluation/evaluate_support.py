
from src.tools.generate_answer import AnswerGenerator


def main():
    generator = AnswerGenerator()

    test_cases = [
        {
            "question": "How can I cancel my order?",
            "expected_intent": "cancel_order",
        },
        {
            "question": "How can I reset my password?",
            "expected_intent": "recover_password",
        },
        {
            "question": "My payment failed but money was deducted.",
            "expected_intent": "payment_issue",
        },
        {
            "question": "My delivery is late.",
            "expected_intent": "delayed_delivery",
        },
        {
            "question": "I want to speak to a human agent.",
            "expected_intent": "contact_human_agent",
        },
    ]

    correct = 0

    print("\nCUSTOMER SUPPORT RETRIEVAL EVALUATION")
    print("=" * 65)

    for case in test_cases:
        result = generator.answer(case["question"])
        sources = result.get("sources", [])

        predicted_intent = (
            sources[0].get("intent")
            if sources
            else None
        )

        is_correct = predicted_intent == case["expected_intent"]
        correct += int(is_correct)

        print(f"\nQuestion: {case['question']}")
        print(f"Expected intent: {case['expected_intent']}")
        print(f"Predicted intent: {predicted_intent}")
        print(f"Result: {'PASS' if is_correct else 'FAIL'}")

    accuracy = correct / len(test_cases) * 100

    print("\n" + "=" * 65)
    print(f"Correct matches: {correct}/{len(test_cases)}")
    print(f"Intent match rate: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
