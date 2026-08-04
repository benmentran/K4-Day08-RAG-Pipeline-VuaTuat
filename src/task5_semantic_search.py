"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


def _generate_hypothetical_doc(query: str) -> str:
    """
    Tạo hypothetical document cho HyDE.

    HyDE thường dùng LLM để sinh một câu trả lời giả định rồi embed câu trả lời đó
    thay vì embed trực tiếp query ngắn. Ở bài lab này ta dùng template ổn định,
    không cần API key, nhưng vẫn đưa query vào ngữ cảnh dạng tài liệu/câu trả lời.
    """
    normalized_query = " ".join(query.split())
    return (
        "Tài liệu liên quan có nội dung trả lời trực tiếp cho câu hỏi sau: "
        f"{normalized_query}. "
        "Nội dung có thể bao gồm quy định, chính sách, điều kiện áp dụng, "
        "quy trình thực hiện, quyền lợi và nghĩa vụ của các bên liên quan."
    )


@lru_cache(maxsize=1)
def _get_embedding_model() -> Any:
    """Load embedding model giống cấu hình Task 4."""
    try:
        from .task4_chunking_indexing import get_embedding_model

        return get_embedding_model()
    except (ImportError, AttributeError):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL)


def _embed_text(text: str) -> list[float]:
    """Embed một văn bản, ưu tiên helper provider-aware nếu Task 4 có định nghĩa."""
    try:
        from .task4_chunking_indexing import embed_texts

        return embed_texts([text])[0]
    except (ImportError, AttributeError):
        model = _get_embedding_model()
        embedding = model.encode(text)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


def _get_collection() -> Any:
    """Lấy ChromaDB collection đã index ở Task 4."""
    try:
        from .task4_chunking_indexing import get_collection

        return get_collection()
    except (ImportError, AttributeError):
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def _distance_to_similarity(distance: float) -> float:
    """Chroma cosine distance -> cosine similarity score."""
    return max(0.0, min(1.0, 1.0 - float(distance)))


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not query or not query.strip() or top_k <= 0:
        return []

    query_text = _generate_hypothetical_doc(query)
    try:
        collection = _get_collection()
        query_vector = _embed_text(query_text)
    except ImportError:
        # Cho phép test/import chạy khi môi trường chưa cài chromadb hoặc model.
        # Sau khi cài requirements.txt và index Task 4, nhánh này sẽ không chạy.
        return []

    requested = min(top_k, max(1, int(top_k)))
    try:
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=requested,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        message = str(exc).lower()
        if "does not contain" in message or "no index" in message or "empty" in message:
            return []
        raise

    documents = results.get("documents", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    output = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        output.append(
            {
                "content": doc,
                "score": round(_distance_to_similarity(dist), 4),
                "metadata": meta or {},
            }
        )

    output.sort(key=lambda item: item["score"], reverse=True)
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")