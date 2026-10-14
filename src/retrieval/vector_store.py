
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.ingestion.chunking import chunk_dataset
from src.ingestion.load_data import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = PROJECT_ROOT / "artifacts" / "vector_store"
BATCH_SIZE = 500


def build_vector_store() -> FAISS:
    print("Loading Bitext dataset...")
    df = load_dataset()

    print("\nCreating overlapping chunks...")
    chunks = chunk_dataset(df)

    documents = [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"],
        )
        for chunk in chunks
    ]

    print(f"\nTotal chunks: {len(documents)}")
    print(f"Embedding in batches of {BATCH_SIZE}...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )

    vector_store = None

    for start in range(0, len(documents), BATCH_SIZE):
        batch = documents[start : start + BATCH_SIZE]
        batch_number = start // BATCH_SIZE + 1
        total_batches = (
            len(documents) + BATCH_SIZE - 1
        ) // BATCH_SIZE

        print(
            f"Processing batch {batch_number}/{total_batches} "
            f"({len(batch)} documents)..."
        )

        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embeddings)
        else:
            vector_store.add_documents(batch)

        # Save progress after every batch.
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(VECTOR_STORE_DIR))

        print(f"Saved progress: {start + len(batch)}/{len(documents)}")

    print(f"\nFAISS vector store saved to: {VECTOR_STORE_DIR}")
    print(f"Documents indexed: {vector_store.index.ntotal}")

    return vector_store


if __name__ == "__main__":
    build_vector_store()
