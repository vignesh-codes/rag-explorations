import requests
import json

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"

# Example text to embed
text_to_embed = "Fault diagnosis in control systems requires understanding system logs and documentation."

# Build payload
payload = {
    "model": MODEL_NAME,
    "prompt": text_to_embed
}

# Call Ollama /api/embeddings endpoint
response = requests.post(OLLAMA_URL, json=payload)

# Check result
if response.status_code == 200:
    data = response.json()
    embedding = data.get("embedding", [])
    print(f"✅ Embedding vector (len={len(embedding)}):")
    print(embedding[:10], "...")  # Print first 10 dims
else:
    print(f"❌ Error {response.status_code}: {response.text}")
