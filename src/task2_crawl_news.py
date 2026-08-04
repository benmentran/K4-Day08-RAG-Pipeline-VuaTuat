"""
Task 2 — Crawl bài viết/hướng dẫn.

Yêu cầu:
    1. Crawl tối thiểu 5 bài viết.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # cần nếu muốn Crawl4AI chạy bằng browser
"""

import asyncio
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://www.ieltsadvantage.com/2023/01/15/ielts-writing-task-2-sample-essays/",
    # "https://www.ieltsbuddy.com/ielts-band-4-essay-samples.html",
    # "https://www.ieltsbuddy.com/ielts-band-5-essay-samples.html",
    # "https://www.ieltsbuddy.com/ielts-band-6-essay-samples.html",
    "https://www.ieltsbuddy.com/ielts-band-7.html",
    "https://www.ieltsbuddy.com/ielts-essay.html",
    "https://www.ieltsbuddy.com/transitional-phrases-for-essays.html",
    "https://www.ieltsbuddy.com/problem-solution-essays.html",
    "https://www.ieltsbuddy.com/ielts-opinion-essays.html",
    "https://www.ieltsbuddy.com/advantage-disadvantage-essay.html",
    "https://www.ieltsbuddy.com/ielts-music-essay.html",
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
}

MIN_CONTENT_CHARS = 500
HTTP_TIMEOUT_SECONDS = 45
CRAWL4AI_TIMEOUT_SECONDS = 8


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    date_crawled = datetime.now().isoformat(timespec="seconds")
    errors: list[str] = []

    try:
        article = await asyncio.to_thread(_crawl_with_requests, url, date_crawled)
        if _has_enough_content(article):
            return article
        errors.append("requests returned too little content")
    except Exception as exc:
        errors.append(f"requests failed: {type(exc).__name__}: {exc}")

    try:
        article = await asyncio.wait_for(
            _crawl_with_crawl4ai(url, date_crawled),
            timeout=CRAWL4AI_TIMEOUT_SECONDS,
        )
        if _has_enough_content(article):
            if errors:
                article["warnings"] = errors
            return article
        errors.append("crawl4ai returned too little content")
    except Exception as exc:  # Depends on Playwright browser/network setup.
        errors.append(f"crawl4ai failed: {type(exc).__name__}: {exc}")

    article = _failed_article(url, date_crawled)
    article["warnings"] = errors
    return article


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filename = f"{i:02d}_{slugify(article['title'])}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath} ({article.get('crawl_status', 'unknown')})")


async def _crawl_with_crawl4ai(url: str, date_crawled: str) -> dict:
    """Crawl bằng Crawl4AI khi Playwright/Chromium đã được cài."""
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    metadata = getattr(result, "metadata", None) or {}
    markdown = _extract_markdown(result).strip()
    title = _clean_text(metadata.get("title") or _title_from_markdown(markdown) or "Untitled article")

    return _make_article(url, title, date_crawled, "crawl4ai", markdown)


def _crawl_with_requests(url: str, date_crawled: str) -> dict:
    """Fallback cho các trang HTML tĩnh hoặc khi Playwright thiếu browser."""
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url, headers=REQUEST_HEADERS, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    content_root = soup.find("article") or soup.find("main") or soup.body or soup
    markdown = _html_to_markdown(content_root)

    return _make_article(
        url, _extract_html_title(soup, content_root), date_crawled,
        "requests+beautifulsoup", markdown,
    )


def _make_article(url: str, title: str, date_crawled: str,
                  crawler: str, content_markdown: str,
                  crawl_status: str = "success") -> dict:
    content_markdown = content_markdown.strip()
    return {
        "url": url,
        "source_domain": urlparse(url).netloc,
        "title": title,
        "date_crawled": date_crawled,
        "crawler": crawler,
        "crawl_status": crawl_status,
        "content_markdown": content_markdown,
        "content_length": len(content_markdown),
    }


def _extract_markdown(result) -> str:
    markdown = getattr(result, "markdown", "") or ""
    if not isinstance(markdown, str):
        markdown = (getattr(markdown, "fit_markdown", None)
                    or getattr(markdown, "raw_markdown", None)
                    or str(markdown))
    return str(markdown)


def _html_to_markdown(root) -> str:
    lines: list[str] = []
    for element in root.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
        text = _clean_text(element.get_text(" ", strip=True))
        if not text:
            continue
        prefix = {"h1": "# ", "h2": "## ", "h3": "### ", "li": "- "}.get(
            element.name, ""
        )
        lines.append(prefix + text)
    return "\n\n".join(lines).strip()


def _extract_html_title(soup, content_root) -> str:
    candidates = []
    for selector in ("h1", "h2", "h3"):
        candidates.extend(
            _clean_text(node.get_text(" ", strip=True))
            for node in content_root.find_all(selector)
        )
    if soup.title:
        candidates.append(_clean_text(soup.title.get_text(" ", strip=True)))

    generic_phrases = ("help center", "home page", "menu", "navigation")
    for candidate in candidates:
        normalized = _normalize_for_match(candidate)
        if candidate and not any(phrase in normalized for phrase in generic_phrases):
            return candidate
    return candidates[0] if candidates else "Untitled article"


def _failed_article(url: str, date_crawled: str) -> dict:
    path_name = urlparse(url).path.strip("/").split("/")[-1] or urlparse(url).netloc
    title = path_name.replace(".html", "").replace("-", " ").title()
    content = (
        "Live content was not available for this URL during crawling. "
        "See warnings for the Crawl4AI and HTTP errors captured from this run."
    )
    return _make_article(
        url, title, date_crawled, "none", f"# {title}\n\n{content}", "failed"
    )


def _has_enough_content(article: dict) -> bool:
    return len(article.get("content_markdown", "").strip()) >= MIN_CONTENT_CHARS


def _title_from_markdown(markdown: str) -> str:
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return _clean_text(text).lower()


def slugify(text: str, max_length: int = 60) -> str:
    text = _normalize_for_match(text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")[:max_length].strip("-")
    return text or "article"


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
    else:
        asyncio.run(crawl_all())
