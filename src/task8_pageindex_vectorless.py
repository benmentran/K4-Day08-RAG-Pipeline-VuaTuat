"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import os
import json
import time
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
_DOC_IDS_FILE = STANDARDIZED_DIR / ".pageindex_doc_ids.json"


def _client():
    """Create the SDK client lazily, so importing this module needs no API key."""
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("PAGEINDEX_API_KEY chưa được cấu hình trong .env")
    try:
        from pageindex.client import PageIndexClient
    except ImportError as exc:
        raise RuntimeError("Chưa cài PageIndex SDK: pip install pageindex") from exc
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _as_dict(response):
    if hasattr(response, "json"):
        response = response.json()
    if not isinstance(response, dict):
        raise RuntimeError(f"PageIndex trả về response không hợp lệ: {response!r}")
    return response


def _pdf_from_markdown(md_file: Path):
    """Create a temporary, readable PDF for the cloud API's document upload."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    text = md_file.read_text(encoding="utf-8")
    # Core PDF fonts are not Unicode fonts; retain content instead of failing on
    # Vietnamese text. A user-provided Unicode font can be added later if needed.
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    pdf.multi_cell(0, 5, text)
    handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    handle.close()
    pdf.output(handle.name)
    return Path(handle.name)


def _id_from(response, *keys):
    response = _as_dict(response)
    for key in keys:
        value = response.get(key)
        if value:
            return value
    raise RuntimeError(f"Không tìm thấy {keys} trong response PageIndex: {response}")


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    client = _client()
    documents = {}
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        pdf_path = _pdf_from_markdown(md_file)
        try:
            response = client.submit_document(str(pdf_path))
            documents[str(md_file.relative_to(STANDARDIZED_DIR))] = _id_from(response, "doc_id", "document_id", "id")
            print(f"  ✓ Uploaded: {md_file.name} -> {documents[str(md_file.relative_to(STANDARDIZED_DIR))]}")
        finally:
            pdf_path.unlink(missing_ok=True)
    _DOC_IDS_FILE.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")
    return documents


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not query or top_k <= 0:
        return []
    if not _DOC_IDS_FILE.exists():
        return []
    try:
        doc_ids = json.loads(_DOC_IDS_FILE.read_text(encoding="utf-8"))
        client = _client()
    except (OSError, json.JSONDecodeError, RuntimeError):
        # PageIndex is an optional remote fallback. A missing key, SDK, or
        # upload cache must not break the local hybrid retrieval pipeline.
        return []
    if not isinstance(doc_ids, dict) or not doc_ids:
        return []
    results = []
    for name, doc_id in doc_ids.items():
        submitted = _as_dict(client.submit_query(doc_id=doc_id, query=query))
        retrieval_id = _id_from(submitted, "retrieval_id", "id")
        deadline = time.monotonic() + float(os.getenv("PAGEINDEX_TIMEOUT", "120"))
        while True:
            retrieval = _as_dict(client.get_retrieval(retrieval_id))
            status = str(retrieval.get("status", "completed")).lower()
            if status in {"completed", "complete", "succeeded", "success"}:
                break
            if status in {"failed", "error", "cancelled", "canceled"}:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"PageIndex retrieval timeout: {retrieval_id}")
            time.sleep(float(os.getenv("PAGEINDEX_POLL_INTERVAL", "1")))
        for rank, node in enumerate(retrieval.get("retrieved_nodes", []), 1):
            for group in node.get("relevant_contents", []):
                for item in group:
                    content = item.get("relevant_content", "").strip()
                    if content:
                        results.append({"content": content, "score": 1.0 / rank,
                                        "metadata": {"section": item.get("section_title"), "document": name},
                                        "source": "pageindex"})
    return results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
