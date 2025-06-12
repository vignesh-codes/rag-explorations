import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from constants import CHROMADB_PATH

DATA_PATH = "data"
COLLECTION_NAME = "test"
BATCH_SIZE = 100

def main():
    # CLI with subcommands
    parser = argparse.ArgumentParser(description="Manage the Chroma DB.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_reset = subparsers.add_parser("reset", help="Reset the database.")
    parser_populate = subparsers.add_parser("populate", help="Populate the database with documents.")
    parser_list = subparsers.add_parser("list", help="List documents in the database.")

    args = parser.parse_args()

    if args.command == "reset":
        clear_database()
    elif args.command == "populate":
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
    elif args.command == "list":
        list_documents()


def load_documents():
    print("📚 Loading documents...")
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = document_loader.load()
    print(f"✅ Loaded {len(documents)} documents.")
    return documents


def split_documents(documents: list[Document]):
    print("🔪 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""],  # Hierarchical splitting
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    print("🗂️  Connecting to Chroma DB...")
    db = Chroma(
        persist_directory=CHROMADB_PATH,
        embedding_function=get_embedding_function(),
        collection_name=COLLECTION_NAME
    )

    # Calculate Page IDs and enrich metadata.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Retrieve existing IDs to avoid duplicates.
    existing_items = db.get()
    existing_ids = set(existing_items["ids"])
    # existing_ids = set(existing_items["ids"])
    print(f"📦 Existing documents in DB: {len(existing_ids)}")

    # Filter new chunks.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding {len(new_chunks)} new chunks (batched)...")
        for i in range(0, len(new_chunks), BATCH_SIZE):
            batch_chunks = new_chunks[i:i + BATCH_SIZE]
            batch_ids = [chunk.metadata["id"] for chunk in batch_chunks]
            print(f"📝 Adding batch {i // BATCH_SIZE + 1} with {len(batch_chunks)} chunks...")
            db.add_documents(batch_chunks, ids=batch_ids)
            # db.persist()
        print("✅ Successfully added all new chunks.")
    else:
        print("✅ No new documents to add.")


def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        # Normalize source path
        source = chunk.metadata.get("source", "")
        source_filename = os.path.basename(source)
        chunk.metadata["source"] = source_filename

        page = chunk.metadata.get("page")
        current_page_id = f"{source_filename}:{page}"

        # Increment chunk index per page
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Assign chunk ID
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add useful metadata
        chunk.metadata["id"] = chunk_id
        chunk.metadata["document_title"] = source_filename
        chunk.metadata["chunk_index"] = current_chunk_index

    return chunks


def list_documents():
    print("🗂️  Listing documents in Chroma DB...")
    db = Chroma(
        persist_directory=CHROMADB_PATH,
        embedding_function=get_embedding_function(),
        collection_name=COLLECTION_NAME
    )
    
    # Do NOT pass include=["ids"] → just call get()
    existing_items = db.get()
    existing_ids = existing_items["ids"]  # list of IDs
    
    print(f"📦 Total documents in DB: {len(existing_ids)}")
    print("📝 First 10 IDs:", existing_ids[:40])



def clear_database():
    if os.path.exists(CHROMADB_PATH):
        print("🗑️  Clearing Chroma DB...")
        shutil.rmtree(CHROMADB_PATH)
        print("✅ Database cleared.")
    else:
        print("⚠️  No database found to clear.")


if __name__ == "__main__":
    main()
