"""
Embedding Helper - Hỗ trợ nhiều embedding provider

Providers:
- sentence_transformers: Local, không cần API key (mặc định)
- google: Cần GEMINI_API_KEY, dimension 768
- openai: Cần OPENAI_API_KEY, dimension 1536
- mistral: Cần MISTRAL_API_KEY, dimension 1024
"""

import os
from typing import Literal

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()


def get_embedding_provider() -> Literal["sentence_transformers", "google", "openai", "mistral"]:
    """Lấy embedding provider từ .env"""
    return os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")


def get_embedding_dimension(provider: str = None) -> int:
    """Trả về embedding dimension của provider"""
    if provider is None:
        provider = get_embedding_provider()
    
    dimensions = {
        "sentence_transformers": 1024,   # BAAI/bge-m3 (1024) - RECOMMENDED
        "google": 768,                   # text-embedding-004
        "openai": 1536,                  # text-embedding-3-small
        "mistral": 1024,                 # mistral-embed
    }
    return dimensions.get(provider, 1024)


def embed_texts_mistral(texts: list[str]) -> list[list[float]]:
    """
    Embed texts bằng Mistral AI API
    
    API: https://api.mistral.ai/v1/embeddings
    Model: mistral-embed
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in .env")
    
    url = "https://api.mistral.ai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    all_embeddings = []
    
    # Mistral API có limit batch size, gửi từng text
    for text in texts:
        payload = {
            "model": "mistral-embed",
            "input": text
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"Mistral API error: {response.status_code} - {response.text}")
        
        data = response.json()
        embedding = data["data"][0]["embedding"]
        all_embeddings.append(embedding)
    
    return all_embeddings


def embed_texts_openai(texts: list[str]) -> list[list[float]]:
    """
    Embed texts bằng OpenAI API
    
    API: https://api.openai.com/v1/embeddings
    Model: text-embedding-3-small
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env")
    
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        url, 
        headers=headers, 
        json={
            "model": "text-embedding-3-small",
            "input": texts
        },
        timeout=60
    )
    
    if response.status_code != 200:
        raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
    
    data = response.json()
    # Sắp xếp theo index vì API có thể trả về không đúng thứ tự
    sorted_embeddings = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_embeddings]


def embed_texts_google(texts: list[str]) -> list[list[float]]:
    """
    Embed texts bằng Google Gemini API
    
    API: https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:batchEmbedContents?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    
    requests_list = [{"taskType": "RETRIEVAL_DOCUMENT", "content": {"parts": [{"text": text}]}} for text in texts]
    
    response = requests.post(
        url,
        headers=headers,
        json={"requests": requests_list},
        timeout=60
    )
    
    if response.status_code != 200:
        raise Exception(f"Google API error: {response.status_code} - {response.text}")
    
    data = response.json()
    return [item["values"] for item in data["embeddings"]]


def embed_texts_sentence_transformers(texts: list[str], model_name: str = "BAAI/bge-m3") -> list[list[float]]:
    """
    Embed texts bằng sentence-transformers (local)
    
    Models:
    - BAAI/bge-m3: Multilingual, ho tro tieng Viet tot (1024 dim) [RECOMMENDED]
    - sentence-transformers/all-MiniLM-L6-v2: Nhe, nhanh (384 dim)
    
    Cai dat: pip install sentence-transformers
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings.tolist()


def embed_texts(texts: list[str], provider: str = None) -> list[list[float]]:
    """
    Embed texts - dispatch theo provider được chọn
    
    Args:
        texts: Danh sách text cần embed
        provider:Embedding provider (nếu None thì đọc từ .env)
    
    Returns:
        List of embeddings (list[float])
    """
    if provider is None:
        provider = get_embedding_provider()
    
    print(f"[*] Embedding with provider: {provider}")
    
    if provider == "mistral":
        return embed_texts_mistral(texts)
    elif provider == "openai":
        return embed_texts_openai(texts)
    elif provider == "google":
        return embed_texts_google(texts)
    elif provider == "sentence_transformers":
        return embed_texts_sentence_transformers(texts)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


def test_mistral_embedding():
    """Test Mistral embedding"""
    print("\n[*] Testing Mistral embedding...")
    try:
        test_texts = ["Hello world", "Testing embedding"]
        embeddings = embed_texts_mistral(test_texts)
        print(f"[OK] Mistral embedding works! Dimension: {len(embeddings[0])}")
        return True
    except Exception as e:
        print(f"[FAIL] Mistral embedding error: {e}")
        return False


if __name__ == "__main__":
    # Test embedding
    provider = get_embedding_provider()
    print(f"Embedding provider: {provider}")
    print(f"Embedding dimension: {get_embedding_dimension()}")
    
    if provider == "mistral":
        test_mistral_embedding()
