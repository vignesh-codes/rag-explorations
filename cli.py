#!/usr/bin/env python3
# Uncomment to enable tracing

# Must precede any llm module imports
from src.utils.config import LANGTRACE_API_KEY
from langtrace_python_sdk import langtrace
langtrace.init(api_key=LANGTRACE_API_KEY)

import argparse
import sys
from src.models.database import RAGDatabase
from src.models.query_engine import QueryEngine
from src.utils.config import validate_config
from src.utils.logging_config import setup_logging, get_logger

def main():
    parser = argparse.ArgumentParser(description="Simple RAG System")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Database commands
    db_parser = subparsers.add_parser("db", help="Database operations")
    db_subparsers = db_parser.add_subparsers(dest="db_command", required=True)
    
    db_subparsers.add_parser("reset", help="Reset database")
    populate_parser = db_subparsers.add_parser("populate", help="Populate database")
    populate_parser.add_argument("--data-path", help="Path to documents")
    db_subparsers.add_parser("list", help="List documents")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", help="Your question")
    query_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of documents to retrieve")
    query_parser.add_argument("--provider", default="openai", help="Embedding provider")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        validate_config()
        
        if args.command == "db":
            database = RAGDatabase(args.provider if hasattr(args, 'provider') else "openai")
            
            if args.db_command == "reset":
                database.reset()
                print("✅ Database reset")
                
            elif args.db_command == "populate":
                database.populate(args.data_path)
                print("✅ Database populated")
                
            elif args.db_command == "list":
                items = database.list_documents()
                print(f"📚 Database contains {len(items['ids'])} documents")
        
        elif args.command == "query":
            database = RAGDatabase(args.provider)
            query_engine = QueryEngine()
            
            result = query_engine.query(database, args.question, args.top_k)
            
            print(f"\n📘 Answer:")
            print(result["answer"])
            print(f"\n🔍 Sources: {', '.join(set(result['sources']))}")
            print(f"\n⏱️ Response Time: {result['response_time']:.2f}s")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())