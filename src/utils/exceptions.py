"""Simple exceptions for the RAG system."""

class RAGError(Exception):
    """Base exception for RAG system errors."""
    pass

class DatabaseError(RAGError):
    """Database operation failed."""
    pass

class QueryError(RAGError):
    """Query processing failed."""
    pass

class DocumentError(RAGError):
    """Document processing failed."""
    pass