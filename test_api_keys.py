"""
Test API Keys - Kiem tra cac API keys co hoat dong khong
Chay: python test_api_keys.py
"""

import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

def test_openrouter():
    """Test OpenRouter API Key"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "sk-or-v1-...":
        return "SKIP", "OpenRouter API key chua duoc set"
    
    print("\n[*] Testing OpenRouter API...")
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return "[OK]", f"OpenRouter hoat dong! Model: {response.json().get('model', 'N/A')}"
        elif response.status_code == 401:
            return "[FAIL]", "OpenRouter API key khong hop le"
        elif response.status_code == 429:
            return "[RATE_LIMIT]", "OpenRouter dang bi rate limit"
        else:
            return "[FAIL]", f"Loi {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return "[ERROR]", f"Loi ket noi: {str(e)}"


def test_mistral():
    """Test Mistral AI API Key"""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key or api_key == "...":
        return "SKIP", "Mistral API key chua duoc set"
    
    print("\n[*] Testing Mistral AI API...")
    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistral-small-latest",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            model = data.get('model', 'N/A')
            return "[OK]", f"Mistral AI hoat dong! Model: {model}"
        elif response.status_code == 401:
            return "[FAIL]", "Mistral API key khong hop le"
        elif response.status_code == 429:
            return "[RATE_LIMIT]", "Mistral dang bi rate limit"
        else:
            return "[FAIL]", f"Loi {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return "[ERROR]", f"Loi ket noi: {str(e)}"


def test_jina():
    """Test Jina API Key - kiem tra qua embedding"""
    api_key = os.getenv("JINA_API_KEY")
    if not api_key or api_key == "jina_...":
        return "SKIP", "Jina API key chua duoc set"
    
    print("\n[*] Testing Jina API (embedding)...")
    try:
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "jina-embeddings-v3",
                "task": "text-matching",
                "dimensions": 1024,
                "input": "Hello world"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            embedding_dim = len(data.get('data', [{}])[0].get('embedding', []))
            return "[OK]", f"Jina AI hoat dong! Embedding dimension: {embedding_dim}"
        elif response.status_code == 401:
            return "[FAIL]", "Jina API key khong hop le"
        elif response.status_code == 429:
            return "[RATE_LIMIT]", "Jina dang bi rate limit"
        else:
            return "[FAIL]", f"Loi {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return "[ERROR]", f"Loi ket noi: {str(e)}"


def test_openai():
    """Test OpenAI API Key"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "sk-proj-...":
        return "SKIP", "OpenAI API key chua duoc set"
    
    print("\n[*] Testing OpenAI API...")
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return "[OK]", f"OpenAI hoat dong! Model: {response.json().get('model', 'N/A')}"
        elif response.status_code == 401:
            return "[FAIL]", "OpenAI API key khong hop le"
        elif response.status_code == 429:
            return "[RATE_LIMIT]", "OpenAI dang bi rate limit"
        else:
            return "[FAIL]", f"Loi {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return "[ERROR]", f"Loi ket noi: {str(e)}"


def test_gemini():
    """Test Gemini API Key"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "...":
        return "SKIP", "Gemini API key chua duoc set"
    
    print("\n[*] Testing Gemini API (gemini-2.0-flash)...")
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [{"text": "Hello"}]
                }],
                "generationConfig": {"maxOutputTokens": 10}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return "[OK]", "Gemini 1.5 Flash hoat dong!"
        elif response.status_code == 400:
            return "[FAIL]", "Gemini API key khong hop le"
        elif response.status_code == 429:
            return "[RATE_LIMIT]", "Gemini dang bi rate limit"
        else:
            return "[FAIL]", f"Loi {response.status_code}: {response.text[:100]}"
    except Exception as e:
        return "[ERROR]", f"Loi ket noi: {str(e)}"


def test_pageindex():
    """Test PageIndex API Key"""
    api_key = os.getenv("PAGEINDEX_API_KEY")
    if not api_key or api_key == "...":
        return "SKIP", "PageIndex API key chua duoc set"
    
    print("\n[*] Testing PageIndex API...")
    try:
        response = requests.get(
            f"https://api.pageindex.ai/health",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30
        )
        
        if response.status_code == 200:
            return "[OK]", "PageIndex hoat dong!"
        elif response.status_code == 401:
            return "[FAIL]", "PageIndex API key khong hop le"
        elif response.status_code == 429:
            return "[RATE_LIMIT]", "PageIndex dang bi rate limit"
        else:
            return "[WARN]", f"Co the hoat dong (status {response.status_code})"
    except Exception as e:
        return "[ERROR]", f"Loi ket noi: {str(e)}"


def main():
    print("=" * 60)
    print("API KEY TESTER")
    print("=" * 60)
    
    results = []
    
    # Test cac API keys
    results.append(("OpenRouter", *test_openrouter()))
    results.append(("Mistral AI", *test_mistral()))
    results.append(("Jina AI", *test_jina()))
    results.append(("OpenAI", *test_openai()))
    results.append(("Gemini", *test_gemini()))
    results.append(("PageIndex", *test_pageindex()))
    
    # In ket qua tong hop
    print("\n" + "=" * 60)
    print("KET QUA TONG HOP")
    print("=" * 60)
    
    success_count = 0
    for name, status, message in results:
        print(f"\n{name}:")
        print(f"  {status}")
        if "[OK]" in status:
            success_count += 1
        print(f"  -> {message}")
    
    print("\n" + "=" * 60)
    print(f"Tong: {success_count}/{len(results)} API keys hoat dong")
    print("=" * 60)
    
    # Khuyen nghi
    print("\nKHUYEN NGHI:")
    if any("OpenRouter" in r[0] and "[OK]" in r[1] for r in results):
        print("  [OK] OpenRouter da hoat dong - ban co the bat dau lab!")
    else:
        print("  [WARN] OpenRouter chua hoat dong - kiem tra lai API key")


if __name__ == "__main__":
    main()
