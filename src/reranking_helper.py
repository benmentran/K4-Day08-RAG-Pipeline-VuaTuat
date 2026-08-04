"""
Reranking Helper - Sử dụng Voyage AI Rerank (Cloud API)

Thay thế Jina API bằng Voyage AI rerank-2 (free tier: 1M tokens/tháng)

Ưu điểm:
- Nhanh, chất lượng cao
- Free tier hào phong (1M tokens/tháng)
- Hỗ trợ tiếng Việt

Cài đặt:
    pip install voyageai

API Key:
    - Đăng ký tại: https://voyage.ai/
    - Free tier: 1M tokens/tháng
    - Thêm vào .env: VOYAGE_API_KEY=your_key_here

Task 7 sử dụng RRF (Reciprocal Rank Fusion) với k=60
"""

import os
from typing import Optional

try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False


def get_voyage_client() -> Optional[object]:
    """Lấy Voyage AI client từ API key"""
    if not VOYAGEAI_AVAILABLE:
        return None
    
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        return None
    
    return voyageai.Client(api_key=api_key)


def rerank_voyage(
    query: str,
    documents: list[str],
    model: str = "rerank-2",
    top_n: int | None = None
) -> list[tuple[int, float]]:
    """
    Rerank documents bằng Voyage AI Rerank API.
    
    Args:
        query: Câu truy vấn
        documents: Danh sách documents cần rerank
        model: Voyage rerank model ("rerank-2" hoặc "rerank-2-lite")
        top_n: Số lượng kết quả trả về (None = lấy tất cả)
    
    Returns:
        List of (document_index, relevance_score) được sắp xếp theo score giảm dần
    """
    client = get_voyage_client()
    
    if client is None:
        raise RuntimeError(
            "Voyage AI client not available. "
            "Set VOYAGE_API_KEY in .env or install voyageai: pip install voyageai"
        )
    
    # Gọi Voyage AI rerank
    result = client.rerank(
        query=query,
        documents=documents,
        model=model
    )
    
    # Lấy top_n nếu specified
    results = [(item.index, item.relevance_score) for item in result.results]
    if top_n is not None:
        results = results[:top_n]
    
    return results


def rerank_rrf(
    rankings: list[list[tuple[int, float]]],
    k: int = 60
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion (RRF) - Gộp nhiều rankings thành 1.
    
    Args:
        rankings: List của rankings, mỗi ranking là list of (doc_index, score)
        k: RRF parameter (default 60 theo spec CP3)
    
    Returns:
        List of (doc_index, rrf_score) được sắp xếp theo rrf_score giảm dần
    
    Spec: CP3 yêu cầu k=60 để cân bằng giữa Semantic và BM25
    """
    rrf_scores = {}
    
    for ranking in rankings:
        for rank, (doc_idx, _) in enumerate(ranking, start=1):
            # RRF formula: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + rrf_score
    
    # Sắp xếp theo RRF score giảm dần
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_results


def rerank_fallback(
    query: str,
    documents: list[str],
    top_n: int = 10
) -> list[tuple[int, float]]:
    """
    Rerank với fallback: Ưu tiên Voyage AI, fallback về score thường.
    """
    try:
        return rerank_voyage(query, documents, top_n=top_n)
    except Exception as e:
        print(f"[WARN] Voyage AI failed: {e}, using fallback scoring")
        # Fallback: simple cosine-like scoring (dùng embedding similarity)
        from src.embedding_helper import embed_texts
        
        embeddings = embed_texts([query] + documents)
        query_emb = embeddings[0]
        doc_embs = embeddings[1:]
        
        import numpy as np
        scores = []
        for emb in doc_embs:
            # Cosine similarity đơn giản
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            scores.append(sim)
        
        # Sắp xếp
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [(idx, score) for idx, score in indexed_scores[:top_n]]


def test_voyage_rerank():
    """Test Voyage AI reranking"""
    print("\n[*] Testing Voyage AI Rerank...")
    
    query = "What is the refund policy?"
    documents = [
        "Our refund policy allows returns within 30 days.",
        "Shipping takes 3-5 business days.",
        "We offer free shipping on orders over $50.",
        "For refunds, please contact support with your order ID.",
        "Our store is open Monday to Friday."
    ]
    
    try:
        results = rerank_voyage(query, documents, top_n=3)
        
        print(f"\n[OK] Voyage AI Rerank works!")
        print(f"Query: {query}")
        print(f"\nTop 3 results:")
        for idx, score in results:
            print(f"  [{score:.4f}] {documents[idx][:60]}...")
        
        # Test RRF
        print(f"\n[*] Testing RRF fusion with k=60...")
        
        # Mock second ranking (BM25)
        bm25_ranking = [
            (0, 0.9),
            (3, 0.8),
            (1, 0.7),
        ]
        
        fused = rerank_rrf([results, bm25_ranking], k=60)
        
        print(f"\n[OK] RRF fusion works!")
        print(f"Fused results:")
        for doc_idx, rrf_score in fused[:3]:
            print(f"  [{rrf_score:.4f}] {documents[doc_idx][:60]}...")
        
        return True
        
    except RuntimeError as e:
        print(f"\n[INFO] Voyage AI not configured: {e}")
        print("[INFO] Get free API key at: https://voyage.ai/")
        return False
    except Exception as e:
        print(f"[FAIL] Voyage AI error: {e}")
        return False


if __name__ == "__main__":
    test_voyage_rerank()
