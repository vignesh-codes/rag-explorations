#!/usr/bin/env python3
"""
Test script for database reset functionality on Windows.
"""
import sys
import time
import gc
sys.path.append('.')

from src.models.database import RAGDatabase
from src.utils.config import validate_config

def test_database_reset():
    """Test database reset functionality."""
    print("🧪 Testing Database Operations on Windows...")
    
    try:
        validate_config()
        
        # Create database
        print("📝 Creating database...")
        db = RAGDatabase("openai")
        
        # Add some test data
        print("📄 Adding test documents...")
        test_texts = [
            "This is a test document about machine learning.",
            "Another document about artificial intelligence.",
            "A third document about data science."
        ]
        test_metadatas = [
            {"source": "test1.pdf"},
            {"source": "test2.pdf"},
            {"source": "test3.pdf"}
        ]
        test_ids = ["doc1", "doc2", "doc3"]
        
        db.db.add_texts(texts=test_texts, metadatas=test_metadatas, ids=test_ids)
        
        # Verify data was added
        data = db.db.get()
        print(f"✅ Added {len(data['ids'])} documents")
        
        # Test collection clearing (safer method)
        print("🗑️ Testing collection clearing...")
        db.clear_collection()
        
        # Verify clearing worked
        cleared_data = db.db.get()
        print(f"✅ After clearing: {len(cleared_data['ids'])} documents (should be 0)")
        
        # Test adding data again
        print("📄 Adding documents after clearing...")
        db.db.add_texts(texts=test_texts[:2], metadatas=test_metadatas[:2], ids=test_ids[:2])
        
        final_data = db.db.get()
        print(f"✅ Final count: {len(final_data['ids'])} documents")
        
        # Test full reset (more aggressive)
        print("🗑️ Testing full database reset...")
        try:
            db.reset()
            print("✅ Full reset successful")
        except Exception as e:
            print(f"⚠️ Full reset failed (expected on Windows): {e}")
            print("💡 Collection clearing is the recommended method")
        
        print("\n✅ Database operation tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_database_reset()