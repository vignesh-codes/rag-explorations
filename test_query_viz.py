#!/usr/bin/env python3
"""
Test script for query visualization functionality.
"""
import sys
sys.path.append('.')

from src.models.database import RAGDatabase
from src.models.query_engine import QueryEngine
from src.utils.config import validate_config

def test_query_visualization():
    """Test query with visualization data."""
    print("🧪 Testing Query Visualization...")
    
    try:
        validate_config()
        
        # Create database and query engine
        print("📝 Setting up components...")
        db = RAGDatabase("openai")
        query_engine = QueryEngine()
        
        # Clear and add test data
        try:
            db.clear_collection()
        except:
            pass
        
        print("📄 Adding test documents...")
        test_texts = [
            "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "Deep learning uses neural networks with multiple layers to process data.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information.",
            "Data science combines statistics, programming, and domain expertise."
        ]
        test_metadatas = [
            {"source": "ml_basics.pdf", "page": 1},
            {"source": "deep_learning.pdf", "page": 1},
            {"source": "nlp_guide.pdf", "page": 1},
            {"source": "cv_intro.pdf", "page": 1},
            {"source": "data_science.pdf", "page": 1}
        ]
        test_ids = [f"chunk_{i}" for i in range(len(test_texts))]
        
        db.db.add_texts(texts=test_texts, metadatas=test_metadatas, ids=test_ids)
        print("✅ Documents added")
        
        # Test query
        print("\n🔍 Testing query with retrieval info...")
        query = "What is machine learning?"
        result = query_engine.query(db, query, k=3)
        
        print(f"📝 Answer: {result['answer'][:100]}...")
        print(f"📊 Retrieved docs: {result['retrieved_docs']}")
        print(f"🎯 Avg similarity: {result['avg_similarity']:.3f}")
        
        # Check retrieved docs info
        if 'retrieved_docs_info' in result:
            print(f"📄 Retrieved docs info: {len(result['retrieved_docs_info'])} chunks")
            for i, doc_info in enumerate(result['retrieved_docs_info']):
                print(f"   Chunk {i+1}: Score {doc_info['similarity_score']:.3f}, Source: {doc_info['metadata'].get('source')}")
        else:
            print("❌ No retrieved_docs_info found")
        
        # Test embeddings retrieval
        print("\n🎯 Testing embeddings retrieval...")
        data = db.db.get(include=['embeddings', 'documents', 'metadatas'])
        embeddings = data.get('embeddings')
        
        if embeddings is not None:
            print(f"✅ Found {len(embeddings)} embeddings")
            print(f"   Embedding dimension: {len(embeddings[0])}")
        else:
            print("❌ No embeddings found")
        
        # Clean up
        db.clear_collection()
        print("\n🧹 Cleaned up test data")
        
        print("\n✅ Query visualization test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_query_visualization()