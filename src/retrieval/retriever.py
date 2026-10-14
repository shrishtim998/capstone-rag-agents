
from pathlib import Path
import re

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = PROJECT_ROOT / "artifacts" / "vector_store"

TOP_K = 4


class HybridRetriever:
    def __init__(self, top_k: int = TOP_K):
        self.top_k = top_k

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = FAISS.load_local(
            str(VECTOR_STORE_DIR),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        # Retrieve documents from the existing FAISS index.
        self.documents = list(
            self.vector_store.docstore._dict.values()
        )

        tokenized_documents = [
            self._tokenize(doc.page_content)
            for doc in self.documents
        ]
        self.bm25 = BM25Okapi(tokenized_documents)

        print(f"Loaded FAISS documents: {len(self.documents)}")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def retrieve(self, query: str) -> list[dict]:
        # Semantic retrieval.
        semantic_docs = self.vector_store.similarity_search(
            query, k=self.top_k
        )

        # Keyword retrieval.
        query_tokens = self._tokenize(query)
        keyword_scores = self.bm25.get_scores(query_tokens)

        keyword_indices = sorted(
            range(len(keyword_scores)),
            key=lambda i: keyword_scores[i],
            reverse=True,
        )[: self.top_k]

        keyword_docs = [
            self.documents[i] for i in keyword_indices
        ]

        # Reciprocal Rank Fusion (RRF).
        fused_scores = {}
        documents_by_key = {}

        for ranked_docs in [semantic_docs, keyword_docs]:
            for rank, doc in enumerate(ranked_docs, start=1):
                key = (
                    doc.metadata.get("record_id"),
                    doc.page_content,
                )
                fused_scores[key] = fused_scores.get(key, 0) + (
                    1 / (60 + rank)
                )
                documents_by_key[key] = doc

        ranked_keys = sorted(
            fused_scores,
            key=fused_scores.get,
            reverse=True,
        )[: self.top_k]

        return [
            {
                "text": documents_by_key[key].page_content,
                "metadata": documents_by_key[key].metadata,
                "score": fused_scores[key],
            }
            for key in ranked_keys
        ]


if __name__ == "__main__":
    retriever = HybridRetriever()

    test_queries = [
        "How can I cancel my order?",
        "How do I get a refund?",
        "My payment failed but money was deducted.",
        "My order has not arrived yet.",
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")

        for result in retriever.retrieve(query):
            print(f"\nMetadata: {result['metadata']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Text: {result['text'][:500]}")
