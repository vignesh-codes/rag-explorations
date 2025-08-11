#!/usr/bin/env python3
"""
Test script for diverse search functionality.
"""
import sys
sys.path.append('.')

from src.models.database import RAGDatabase
from src.models.query_engine import QueryEngine
from src.utils.config import validate_config

def test_diverse_search():
    """Test diverse search vs regular search."""
    print("🧪 Testing Diverse Search...")
    
    try:
        validate_config()
        
        # Create database and query engine
        db = RAGDatabase("openai")
        query_engine = QueryEngine()
        
        # Test query about gross margins
        query = "What are Intel gross margins?"
        
        print(f"\n❓ Query: {query}")
        
        # Test regular search
        print("\n📊 Regular Search Results:")
        regular_results = db.search(query, k=5)
        print(f"Retrieved {len(regular_results)} chunks:")
        for i, (doc, score) in enumerate(regular_results):
            source = doc.metadata.get('source', 'Unknown')
            print(f"  {i+1}. {source} (score: {score:.3f})")
        
        # Test diverse search
        print("\n🎯 Diverse Search Results:")
        diverse_results = db.search_diverse(query, k=5, min_sources=2)
        print(f"Retrieved {len(diverse_results)} chunks:")
        sources_found = set()
        for i, (doc, score) in enumerate(diverse_results):
            source = doc.metadata.get('source', 'Unknown')
            sources_found.add(source)
            print(f"  {i+1}. {source} (score: {score:.3f})")
        
        print(f"\n📈 Source Diversity:")
        print(f"Regular search sources: {len(set(doc.metadata.get('source', 'Unknown') for doc, _ in regular_results))}")
        print(f"Diverse search sources: {len(sources_found)}")
        
        # Test full query with diverse search
        print(f"\n🤖 Full Query Response:")
        result = query_engine._process_results(diverse_results, query)
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources used: {set(result['sources'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_diverse_search()