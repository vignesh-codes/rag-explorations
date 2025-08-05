#!/usr/bin/env python3
"""
Test script to check ChromaDB embeddings storage and retrieval.
"""
import sys
import numpy as np
sys.path.append('.')

from src.models.database import RAGDatabase
from src.utils.config import validate_config

def test_embeddings():
    """Test embeddings storage and retrieval."""
    print("🧪 Testing ChromaDB Embeddings...")
    
    try:
        validate_config()
        
        # Create database
        print("📝 Creating database...")
        db = RAGDatabase("openai")
        
        # Clear any existing data
        try:
            db.clear_collection()
        except:
            pass
        
        # Add some test data
        print("📄 Adding test documents...")
        test_texts = [
            "This is a test document about machine learning and artificial intelligence.",
            "Another document discussing data science and analytics.",
            "A third document about natural language processing."
        ]
        test_metadatas = [
            {"source": "test1.pdf", "page": 1},
            {"source": "test2.pdf", "page": 1},
            {"source": "test3.pdf", "page": 1}
        ]
        test_ids = ["doc1", "doc2", "doc3"]
        
        db.db.add_texts(texts=test_texts, metadatas=test_metadatas, ids=test_ids)
        print("✅ Documents added")
        
        # Test different ways to get data
        print("\n🔍 Testing data retrieval methods...")
        
        # Method 1: Default get()
        print("1. Default get():")
        data1 = db.db.get()
        print(f"   Keys: {list(data1.keys())}")
        print(f"   Documents: {len(data1.get('documents', []))}")
        embeddings1 = data1.get('embeddings')
        print(f"   Embeddings: {len(embeddings1) if embeddings1 else 'None'}")
        
        # Method 2: Explicit include
        print("\n2. get(include=['embeddings', 'documents', 'metadatas']):")
        data2 = db.db.get(include=['embeddings', 'documents', 'metadatas'])
        print(f"   Keys: {list(data2.keys())}")
        print(f"   Documents: {len(data2.get('documents', []))}")
        embeddings2 = data2.get('embeddings')
        print(f"   Embeddings: {len(embeddings2) if embeddings2 is not None else 'None'}")
        
        # Method 3: Check if embeddings exist and their shape
        embeddings = data2.get('embeddings')
        if embeddings is not None and len(embeddings) > 0:
            print(f"   Embedding type: {type(embeddings)}")
            print(f"   First embedding type: {type(embeddings[0])}")
            print(f"   First embedding shape: {np.array(embeddings[0]).shape}")
        else:
            print("   No embeddings found")
        
        # Method 4: Try similarity search to see if embeddings work
        print("\n3. Testing similarity search:")
        results = db.search("machine learning", k=2)
        print(f"   Search results: {len(results)} documents found")
        for doc, score in results:
            print(f"   - Score: {score:.3f}, Text: {doc.page_content[:50]}...")
        
        # Clean up
        db.clear_collection()
        print("\n🧹 Cleaned up test data")
        
        print("\n✅ Embeddings test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_embeddings()