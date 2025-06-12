from langchain_ollama import OllamaEmbeddings
from constants import EMBEDDING_MODEL


def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="http://localhost:11434",
    )
    return embeddings
