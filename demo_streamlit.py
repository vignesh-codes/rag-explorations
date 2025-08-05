#!/usr/bin/env python3
"""
Demo script to test the Streamlit PDF Chat app functionality.
"""
import os
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.append('.')

from src.models.database import RAGDatabase
from src.models.document_processor import process_documents
from src.models.query_engine import QueryEngine
from src.utils.config import validate_config
from langchain_community.document_loaders import PyPDFLoader

def create_sample_pdf():
    """Create a sample PDF for testing (if you don't have one)."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create a temporary PDF
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        
        # Create PDF content
        c = canvas.Canvas(temp_pdf.name, pagesize=letter)
        
        # Add some sample content
        c.drawString(100, 750, "Sample Document for RAG Testing")
        c.drawString(100, 700, "")
        c.drawString(100, 650, "This is a sample document created for testing the PDF Chat Assistant.")
        c.drawString(100, 600, "")
        c.drawString(100, 550, "Key Information:")
        c.drawString(120, 500, "• The application deadline is December 15, 2024")
        c.drawString(120, 450, "• Tuition fee is $50,000 per year")
        c.drawString(120, 400, "• Prerequisites include: Mathematics, Physics, Computer Science")
        c.drawString(120, 350, "• The program duration is 4 years")
        c.drawString(120, 300, "• Financial aid is available for qualified students")
        c.drawString(100, 250, "")
        c.drawString(100, 200, "Contact Information:")
        c.drawString(120, 150, "Email: admissions@university.edu")
        c.drawString(120, 100, "Phone: (555) 123-4567")
        
        c.save()
        temp_pdf.close()
        
        return temp_pdf.name
        
    except ImportError:
        print("⚠️  reportlab not installed. Cannot create sample PDF.")
        print("Install with: pip install reportlab")
        return None

def test_rag_system():
    """Test the RAG system components."""
    print("🧪 Testing RAG System Components...")
    
    try:
        # Validate configuration
        validate_config()
        print("✅ Configuration valid")
        
        # Initialize components
        database = RAGDatabase("openai")
        query_engine = QueryEngine()
        print("✅ Components initialized")
        
        # Create or use sample PDF
        sample_pdf = create_sample_pdf()
        if not sample_pdf:
            print("❌ Could not create sample PDF. Please provide a PDF file manually.")
            return False
        
        print(f"📄 Created sample PDF: {sample_pdf}")
        
        # Process PDF
        print("📝 Processing PDF...")
        loader = PyPDFLoader(sample_pdf)
        documents = loader.load()
        
        # Process into chunks
        chunks = process_documents(documents)
        print(f"✅ Extracted {len(chunks)} document chunks from {len(documents)} pages")
        
        # Add to database
        print("💾 Adding to vector database...")
        database.reset()
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        database.db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        print("✅ Document chunks added to database")
        
        # Test queries
        test_queries = [
            "What is the application deadline?",
            "How much is the tuition fee?",
            "What are the prerequisites?",
            "How long is the program?",
            "Is financial aid available?"
        ]
        
        print("\n🔍 Testing queries...")
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            result = query_engine.query(database, query, k=3)
            
            print(f"📝 Answer: {result['answer'][:100]}...")
            print(f"⏱️  Response Time: {result['response_time']:.2f}s")
            print(f"📊 Retrieved Docs: {result['retrieved_docs']}")
            print(f"🎯 Avg Similarity: {result['avg_similarity']:.3f}")
        
        # Clean up
        os.unlink(sample_pdf)
        print(f"\n🧹 Cleaned up sample PDF")
        
        print("\n✅ All tests passed! The Streamlit app should work correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main demo function."""
    print("🚀 PDF Chat Assistant Demo")
    print("=" * 50)
    
    # Test the system
    if test_rag_system():
        print("\n🎉 Demo completed successfully!")
        print("\nNext steps:")
        print("1. Run: python run_streamlit.py")
        print("2. Upload your own PDF files")
        print("3. Start chatting with your documents!")
        print("4. Check the Analytics tab for performance metrics")
    else:
        print("\n❌ Demo failed. Please check your configuration.")
        print("Make sure your .env file has the correct OpenAI API key.")

if __name__ == "__main__":
    main()