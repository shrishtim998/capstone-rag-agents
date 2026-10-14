
from typing import Any

import pandas as pd


def chunk_dataset(
    df: pd.DataFrame,
    chunk_size: int = 120,
    chunk_overlap: int = 20,
) -> list[dict[str, Any]]:
    """Create overlapping word chunks while preserving Bitext metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size"
        )

    chunks = []
    step = chunk_size - chunk_overlap

    for row_id, row in df.reset_index(drop=True).iterrows():
        question = str(row["instruction"]).strip()
        response = str(row["response"]).strip()

        text = (
            f"Customer question: {question}\n"
            f"Support response: {response}"
        )

        words = text.split()

        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size]

            if not chunk_words:
                break

            chunks.append(
                {
                    "text": " ".join(chunk_words),
                    "metadata": {
                        "record_id": int(row_id),
                        "category": str(row["category"]),
                        "intent": str(row["intent"]),
                        "source": "bitext",
                    },
                }
            )

            if start + chunk_size >= len(words):
                break

    return chunks


if __name__ == "__main__":
    from src.ingestion.load_data import load_dataset

    dataset = load_dataset()
    chunks = chunk_dataset(dataset)

    print(f"\nTotal chunks created: {len(chunks)}")
    print("\nFirst chunk:")
    print(chunks[0]["text"])
    print("\nFirst chunk metadata:")
    print(chunks[0]["metadata"])
