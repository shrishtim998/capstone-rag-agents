
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "my_support_data.csv"

REQUIRED_COLUMNS = ["instruction", "category", "intent", "response"]


def load_small_dataset() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            "Place my_support_data.csv inside the data folder."
        )

    df = pd.read_csv(CSV_PATH)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["instruction", "response"]).copy()
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df["response"] = df["response"].astype(str).str.strip()
    df = df[
        df["instruction"].ne("") & df["response"].ne("")
    ].reset_index(drop=True)

    print(f"Loaded dataset: {CSV_PATH.name}")
    print(f"Number of support records: {len(df)}")
    return df


class SmallDataSupportBot:
    def __init__(self):
        self.df = load_small_dataset()

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.question_vectors = self.vectorizer.fit_transform(
            self.df["instruction"]
        )

        print("Loading Hugging Face model...")
        self.generator = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=-1,
        )
        print("Small-data support bot ready.")

    def answer(self, question: str) -> dict:
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(
            query_vector, self.question_vectors
        )[0]

        best_index = int(scores.argmax())
        best_record = self.df.iloc[best_index]
        similarity = float(scores[best_index])

        # Avoid using an unrelated record for an unknown question.
        if similarity < 0.10:
            return {
                "question": question,
                "answer": (
                    "I could not find sufficiently relevant information "
                    "in the sample support dataset. Please contact "
                    "customer support for assistance."
                ),
                "matched_question": None,
                "intent": None,
                "similarity": round(similarity, 3),
            }

        prompt = f"""
You are an e-commerce customer support assistant.
Write a helpful answer to the customer using only the support information.
Do not invent policies, dates, fees, or order details.
If the information is insufficient, tell the customer to contact support.
Use clear, complete sentences and preserve the useful steps.

Customer question: {question}

Related support question: {best_record['instruction']}

Support information:
{best_record['response']}

Answer:
""".strip()

        result = self.generator(
            prompt,
            max_new_tokens=120,
            do_sample=False,
            num_beams=4,
            truncation=True,
        )

        generated = result[0]["generated_text"].strip()

        # Keep the retrieved answer if generation is empty or unusable.
        if len(generated.split()) < 8:
            generated = best_record["response"]

        return {
            "question": question,
            "answer": generated,
            "matched_question": best_record["instruction"],
            "intent": best_record["intent"],
            "similarity": round(similarity, 3),
        }


if __name__ == "__main__":
    bot = SmallDataSupportBot()

    test_questions = [
        "How do I cancel an order?",
        "My payment failed but money was deducted.",
        "I forgot my password. How can I get back in?",
        "My package arrived damaged.",
        "How can I download my bill?",
    ]

    for question in test_questions:
        print("\n" + "=" * 65)
        print("QUESTION:", question)

        try:
            result = bot.answer(question)
            print("MATCHED QUESTION:", result["matched_question"])
            print("INTENT:", result["intent"])
            print("SIMILARITY:", result["similarity"])
            print("ANSWER:\n", result["answer"])
        except Exception as exc:
            print(f"ERROR: {type(exc).__name__}: {exc}")
