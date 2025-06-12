import requests

# Config
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "nomic-embed-text"  # or "llama3:70b" if you pulled that one

# Function to send prompt to Ollama
def query_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code == 200:
        data = response.json()
        return data.get("response", "")
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# Example usage
if __name__ == "__main__":
    user_prompt = "what model are you using?"
    answer = query_ollama(user_prompt)
    print(f"\nmistral Answer:\n{answer}")
