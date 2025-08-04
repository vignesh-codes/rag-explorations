"""Simple embedding functions."""

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from .config import OPENAI_API_KEY, EMBEDDING_MODEL

def get_openai_embeddings():
    """Get OpenAI embeddings."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

def get_ollama_embeddings():
    """Get Ollama embeddings."""
    return OllamaEmbeddings(model="nomic-embed-text")

def get_embeddings(provider="openai"):
    """Get embeddings based on provider."""
    if provider == "openai":
        return get_openai_embeddings()
    elif provider == "ollama":
        return get_ollama_embeddings()
    else:
        raise ValueError(f"Unknown provider: {provider}")