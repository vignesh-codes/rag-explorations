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
        self._client = None
    
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
    
    def clear_collection(self):
        """Clear all documents from the collection without deleting files."""
        try:
            # Get all document IDs
            data = self.db.get()
            if data['ids']:
                # Delete all documents
                self.db.delete(ids=data['ids'])
            return True
        except Exception as e:
            raise DatabaseError(f"Failed to clear collection: {e}")
    
    def reset(self):
        """Reset database with Windows-friendly approach."""
        import time
        import gc
        
        # Close existing connection first
        if self._db is not None:
            try:
                # Try to delete the collection first
                self._db.delete_collection()
            except:
                pass
            self._db = None
        
        # Force garbage collection to release references
        gc.collect()
        time.sleep(1.0)  # Give Windows time to release file handles
        
        # Try multiple approaches to remove the database
        if self.db_path.exists():
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    shutil.rmtree(self.db_path)
                    break  # Success!
                except PermissionError as e:
                    if attempt < max_attempts - 1:
                        # Wait longer and try again
                        time.sleep(2.0)
                        gc.collect()
                    else:
                        # Last attempt failed - try alternative approach
                        try:
                            self._force_remove_directory(self.db_path)
                        except Exception:
                            # If all else fails, create a new database path
                            import uuid
                            new_path = self.db_path.parent / f"chroma_db_{uuid.uuid4().hex[:8]}"
                            self.db_path = new_path
                            break
        
        self._db = None
    
    def _force_remove_directory(self, path):
        """Force remove directory on Windows using alternative methods."""
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            try:
                # Use Windows rmdir command with force
                subprocess.run(['rmdir', '/s', '/q', str(path)], 
                             shell=True, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                # If that fails, try robocopy trick (Windows-specific)
                empty_dir = path.parent / "empty_temp"
                empty_dir.mkdir(exist_ok=True)
                try:
                    subprocess.run(['robocopy', str(empty_dir), str(path), '/mir'], 
                                 shell=True, capture_output=True)
                    shutil.rmtree(path)
                    shutil.rmtree(empty_dir)
                except:
                    # Clean up empty dir if it exists
                    if empty_dir.exists():
                        try:
                            shutil.rmtree(empty_dir)
                        except:
                            pass
                    raise
        else:
            # Non-Windows systems
            shutil.rmtree(path)
    
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
    
    def search_diverse(self, query, k=5, min_sources=2):
        """Search for similar documents ensuring diversity across sources."""
        # Get more results initially
        initial_results = self.db.similarity_search_with_score(query, k=k*2)
        
        if not initial_results:
            return []
        
        # Group by source
        source_groups = {}
        for doc, score in initial_results:
            source = doc.metadata.get('source', 'unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append((doc, score))
        
        # Ensure we have at least min_sources represented
        diverse_results = []
        sources_used = set()
        
        # First pass: get best result from each source
        for source, docs in source_groups.items():
            if len(diverse_results) < k:
                best_doc, best_score = min(docs, key=lambda x: x[1])  # Lower score = better
                diverse_results.append((best_doc, best_score))
                sources_used.add(source)
        
        # Second pass: fill remaining slots with best remaining results
        remaining_docs = []
        for source, docs in source_groups.items():
            for doc, score in docs:
                if (doc, score) not in diverse_results:
                    remaining_docs.append((doc, score))
        
        # Sort remaining by score and add until we reach k
        remaining_docs.sort(key=lambda x: x[1])
        for doc, score in remaining_docs:
            if len(diverse_results) < k:
                diverse_results.append((doc, score))
        
        # Sort final results by score
        diverse_results.sort(key=lambda x: x[1])
        return diverse_results[:k]
    
    def list_documents(self):
        """List all documents."""
        return self.db.get()
    
    def close(self):
        """Close database connection."""
        if self._db is not None:
            try:
                # ChromaDB doesn't have an explicit close method,
                # but we can clear the reference
                self._db = None
            except:
                pass