"""Simple database operations."""

import os
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from ..utils.config import DB_PATH, DATA_PATH
from ..utils.embeddings import get_embeddings
from ..utils.exceptions import DatabaseError
from .document_processor import process_documents

class RAGDatabase:
    """Simple RAG database."""
    
    def __init__(self, embedding_provider="openai"):
        self.embedding_function = get_embeddings(embedding_provider)
        self.db_path = Path(DB_PATH)
        self.collection_name = "documents"
        self._db = None
    
    @property
    def db(self):
        """Get database connection."""
        if self._db is None:
            self._db = Chroma(
                persist_directory=str(self.db_path),
                embedding_function=self.embedding_function,
                collection_name=self.collection_name,
            )
        return self._db
    
    def reset(self):
        """Reset database."""
        if self.db_path.exists():
            shutil.rmtree(self.db_path)
        self._db = None
    
    def populate(self, data_path=None):
        """Load and process documents."""
        data_path = data_path or DATA_PATH
        
        # Load PDFs
        loader = PyPDFDirectoryLoader(data_path)
        documents = loader.load()
        
        # Process documents
        chunks = process_documents(documents)
        
        # Add to database
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        self.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    
    def search(self, query, k=5):
        """Search for similar documents."""
        return self.db.similarity_search_with_score(query, k=k)
    
    def list_documents(self):
        """List all documents."""
        return self.db.get()