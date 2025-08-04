"""Simple configuration for RAG system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Simple configuration variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGTRACE_API_KEY = os.getenv("LANGTRACE_API_KEY")

# Paths
DATA_PATH = "data"
VISUALIZATION_PATH = "visualizations"
DB_PATH = "chroma_db"

# Model settings
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-3.5-turbo"

# Chunking settings
MAX_CHUNK_TOKENS = 300
MIN_CHUNK_TOKENS = 80
OVERLAP_SENTENCES = 1

# Create directories
Path(VISUALIZATION_PATH).mkdir(exist_ok=True)

def validate_config():
    """Check if required configuration is present."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    if not Path(DATA_PATH).exists():
        raise ValueError(f"Data path '{DATA_PATH}' does not exist")