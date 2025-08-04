"""Tests for the RAG system."""

import pytest
from unittest.mock import Mock, patch

from src.config import Config
from src.database import RAGDatabase
from src.document_processor import DocumentProcessor
from src.exceptions import RAGException, DatabaseError, DocumentProcessingError
from src.query import RAGQueryEngine


class TestConfig:
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test config initializes with defaults."""
        config = Config()
        assert config.data_path == "data"
        assert config.database.collection_name == "rag_documents"
        assert config.model.generation_model == "gpt-3.5-turbo"
    
    def test_config_validation_missing_api_key(self):
        """Test config validation fails without API key."""
        config = Config()
        config.openai_api_key = None
        
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            config.validate()


class TestDocumentProcessor:
    """Test document processing functionality."""
    
    def test_token_counting(self):
        """Test token counting method."""
        text = "This is a test sentence with multiple words."
        count = DocumentProcessor._count_tokens(text)
        assert count == 9
    
    def test_chunk_id_generation(self):
        """Test chunk ID generation."""
        from langchain.schema.document import Document
        
        doc = Document(
            page_content="Test content",
            metadata={"source": "/path/to/test.pdf", "page": 1}
        )
        
        chunk_id = DocumentProcessor.generate_chunk_id(doc)
        assert isinstance(chunk_id, str)
        assert len(chunk_id) == 40  # SHA1 hash length
    
    @patch('src.document_processor.PyPDFDirectoryLoader')
    def test_load_documents_success(self, mock_loader):
        """Test successful document loading."""
        mock_loader.return_value.load.return_value = [
            Mock(page_content="Test content", metadata={"source": "test.pdf"})
        ]
        
        processor = DocumentProcessor("test_data")
        documents = processor.load_documents()
        
        assert len(documents) == 1
        mock_loader.assert_called_once()
    
    def test_load_documents_invalid_path(self):
        """Test document loading with invalid path."""
        with pytest.raises(DocumentProcessingError):
            DocumentProcessor("nonexistent_path")


class TestRAGDatabase:
    """Test database operations."""
    
    @patch('src.database.get_embedding_function')
    @patch('src.database.Chroma')
    def test_database_initialization(self, mock_chroma, mock_embedding):
        """Test database initialization."""
        mock_embedding.return_value = Mock()
        
        db = RAGDatabase()
        assert db.collection_name == "rag_documents"
        mock_embedding.assert_called_once_with("openai")
    
    @patch('src.database.shutil.rmtree')
    @patch('src.database.Path.exists')
    def test_reset_database(self, mock_exists, mock_rmtree):
        """Test database reset."""
        mock_exists.return_value = True
        
        with patch('src.database.get_embedding_function'):
            db = RAGDatabase()
            db.reset_database()
        
        mock_rmtree.assert_called_once()


class TestRAGQueryEngine:
    """Test query engine functionality."""
    
    @patch('src.query.ChatOpenAI')
    def test_query_engine_initialization(self, mock_openai):
        """Test query engine initialization."""
        mock_database = Mock()
        
        engine = RAGQueryEngine(mock_database)
        assert engine.database == mock_database
        mock_openai.assert_called_once()
    
    def test_query_no_results(self):
        """Test query with no results."""
        mock_database = Mock()
        mock_database.similarity_search.return_value = []
        
        with patch('src.query.ChatOpenAI'):
            engine = RAGQueryEngine(mock_database)
            result = engine.query("test question")
        
        assert result["answer"] == "I don't know based on the provided context."
        assert result["sources"] == []


class TestIntegration:
    """Integration tests for the RAG system."""
    
    @pytest.fixture
    def mock_environment(self):
        """Set up mock environment for testing."""
        with patch.dict('os.environ', {
            'OPENAI_API_KEY': 'test-key',
            'LANGCHAIN_API_KEY': 'test-key'
        }):
            yield
    
    @patch('src.database.DocumentProcessor')
    @patch('src.database.get_embedding_function')
    @patch('src.database.Chroma')
    def test_end_to_end_workflow(self, mock_chroma, mock_embedding, mock_processor, mock_environment):
        """Test end-to-end RAG workflow."""
        # Mock document processing
        mock_processor.return_value.load_documents.return_value = [
            Mock(page_content="Test content", metadata={"source": "test.pdf"})
        ]
        mock_processor.return_value.split_documents.return_value = [
            Mock(page_content="Test chunk", metadata={"source": "test.pdf", "page": 1})
        ]
        
        # Mock database
        mock_db_instance = Mock()
        mock_chroma.return_value = mock_db_instance
        mock_embedding.return_value = Mock()
        
        # Test database population
        db = RAGDatabase()
        db.populate_database()
        
        # Verify calls were made
        mock_processor.assert_called()
        mock_db_instance.add_texts.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])