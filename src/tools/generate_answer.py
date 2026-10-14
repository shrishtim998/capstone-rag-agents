
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "my_support_data.csv"

REQUIRED_COLUMNS = [
    "instruction",
    "category",
    "intent",
    "response",
]

MIN_SIMILARITY = 0.10

DELIVERY_TERMS = [
    "not arrived",
    "hasn't arrived",
    "has not arrived",
    "not delivered",
    "delivery is late",
    "package is late",
    "where is my order",
    "order is delayed",
    "where is my package",
    "my package is missing",
    "still waiting for my order",
    "my order has not arrived yet",
    "my order has not arrived",
    "package hasn't arrived",
    "package has not arrived",
]


class AnswerGenerator:
    def __init__(self):
        print("Loading the custom support dataset...")

        if not CSV_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found: {CSV_PATH}\n"
                "Place my_support_data.csv inside the data folder."
            )

        self.df = pd.read_csv(CSV_PATH)

        missing = [
            column
            for column in REQUIRED_COLUMNS
            if column not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required CSV columns: {missing}"
            )

        for column in REQUIRED_COLUMNS:
            self.df[column] = (
                self.df[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        self.df = self.df[
            self.df["instruction"].ne("")
            & self.df["response"].ne("")
        ].reset_index(drop=True)

        if self.df.empty:
            raise ValueError(
                "The support dataset contains no valid records."
            )

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )

        self.question_vectors = self.vectorizer.fit_transform(
            self.df["instruction"]
        )

        print(f"Loaded support records: {len(self.df)}")
        print("Custom support answer generator ready.")

    def answer(self, question: str) -> dict:
        question = question.strip()

        if not question:
            return {
                "question": question,
                "answer": "Please enter a customer support question.",
                "sources": [],
                "confidence": "low",
            }

        query_vector = self.vectorizer.transform([question])

        scores = cosine_similarity(
            query_vector,
            self.question_vectors,
        )[0]

        query_lower = question.lower()
        best_index = int(scores.argmax())
        forced_match = False

        # Handle delivery-related questions before checking
        # the similarity threshold.
        is_delivery_question = any(
            term in query_lower for term in DELIVERY_TERMS
        )

        if is_delivery_question:
            delivery_matches = self.df.index[
                self.df["intent"].eq("delayed_delivery")
            ].tolist()

            if delivery_matches:
                best_index = delivery_matches[0]
                forced_match = True

        best_record = self.df.iloc[best_index]
        similarity = float(scores[best_index])

        # Avoid returning an unrelated answer for unsupported questions.
        if (
            similarity < MIN_SIMILARITY
            and not forced_match
        ):
            return {
                "question": question,
                "answer": (
                    "I couldn't find sufficiently relevant information "
                    "in the available support records. Please contact "
                    "customer support for assistance."
                ),
                "sources": [],
                "confidence": "low",
            }

        # Return the full answer stored in the CSV.
        answer_text = best_record["response"]

        source = {
            "record_id": int(best_index),
            "category": best_record["category"],
            "intent": best_record["intent"],
            "source": CSV_PATH.name,
            "matched_question": best_record["instruction"],
            "similarity": round(similarity, 3),
        }

        return {
            "question": question,
            "answer": answer_text,
            "sources": [source],
            "confidence": "unvalidated",
        }


if __name__ == "__main__":
    generator = AnswerGenerator()

    test_questions = [
        "How can I cancel my order?",
        "How do I reset my password?",
        "My payment failed but money was deducted.",
        "My order has not arrived yet.",
        "How can I speak to a human support agent?",
        "My delivery is late. Where is my package?",
        "My package still hasn't arrived.",
        "What is the weather today?",
    ]

    for question in test_questions:
        print("\n" + "=" * 70)
        print("QUESTION:", question)

        try:
            result = generator.answer(question)

            print("\nANSWER:\n", result["answer"])
            print("\nSOURCES:")
            print(json.dumps(result["sources"], indent=2))
            print("\nCONFIDENCE:", result["confidence"])

        except Exception as exc:
            print(f"\nERROR: {type(exc).__name__}: {exc}")
