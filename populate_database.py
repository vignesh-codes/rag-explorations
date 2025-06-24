import argparse
import os
import shutil
import hashlib
import nltk
from nltk.tokenize import sent_tokenize
from itertools import islice
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from constants import CHROMADB_PATH

# One-time download for NLTK
nltk.download("punkt")
nltk.download("punkt_tab")

DATA_PATH = "data"
COLLECTION_NAME = "test"
BATCH_SIZE = 2000  # Can increase based on CPU/memory

def main():
    parser = argparse.ArgumentParser(description="Manage the Chroma DB.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("reset", help="Reset the database.")
    subparsers.add_parser("populate", help="Populate the database with documents.")
    subparsers.add_parser("list", help="List documents in the database.")

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
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents.")
    return documents

def split_documents(documents: list[Document]):
    print("🧠 Performing semantic-aware chunking with sliding window...")

    max_chunk_tokens = 300
    min_chunk_tokens = 80
    overlap_sentences = 1

    def count_tokens(text: str) -> int:
        return len(text.split())  # Simplified, can replace with tokenizer

    final_chunks = []

    for doc in documents:
        metadata = doc.metadata.copy()
        sentences = sent_tokenize(doc.page_content)

        current_chunk = []
        current_token_count = 0
        index = 0

        while index < len(sentences):
            sentence = sentences[index]
            token_count = count_tokens(sentence)

            if current_token_count + token_count <= max_chunk_tokens:
                current_chunk.append(sentence)
                current_token_count += token_count
                index += 1
            else:
                if current_token_count >= min_chunk_tokens:
                    text_chunk = " ".join(current_chunk)
                    final_chunks.append(Document(page_content=text_chunk, metadata=metadata.copy()))
                    current_chunk = current_chunk[-overlap_sentences:]
                    current_token_count = sum(count_tokens(s) for s in current_chunk)
                else:
                    current_chunk.append(sentence)
                    text_chunk = " ".join(current_chunk)
                    final_chunks.append(Document(page_content=text_chunk, metadata=metadata.copy()))
                    current_chunk = []
                    current_token_count = 0
                    index += 1

        if current_chunk:
            text_chunk = " ".join(current_chunk)
            final_chunks.append(Document(page_content=text_chunk, metadata=metadata.copy()))

    print(f"✅ Created {len(final_chunks)} coherent chunks.")
    return final_chunks

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def generate_unique_id(chunk: Document) -> str:
    source = os.path.basename(chunk.metadata.get("source", "unknown"))
    page = str(chunk.metadata.get("page", "0"))
    chunk_text = chunk.page_content
    base = f"{source}|{page}|{chunk_text.strip()[:50]}"  # hash partial text for uniqueness
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while (chunk := list(islice(it, n))):
        yield chunk

def add_to_chroma(chunks: list[Document]):
    print("🗂️ Connecting to Chroma DB...")
    db = Chroma(
        persist_directory=CHROMADB_PATH,
        embedding_function=get_embedding_function(),
        collection_name=COLLECTION_NAME,
    )

    texts = []
    metadatas = []
    ids = []

    for chunk in chunks:
        meta = chunk.metadata.copy()
        source = os.path.basename(meta.get("source", "unknown"))
        page = meta.get("page", "0")
        chunk_index = meta.get("chunk_index", "0")
        base_id = f"{source}:{page}:{chunk_index}"
        unique_id = generate_unique_id(chunk)

        meta["id"] = unique_id
        meta["source"] = source
        meta["chunk_index"] = chunk_index
        meta["document_title"] = source

        texts.append(chunk.page_content)
        metadatas.append(meta)
        ids.append(unique_id)

    print(f"📥 Inserting {len(texts)} chunks to DB in batches of {BATCH_SIZE}...")

    for b_ids, b_texts, b_metas in zip(
        batched(ids, BATCH_SIZE),
        batched(texts, BATCH_SIZE),
        batched(metadatas, BATCH_SIZE),
    ):
        db.add_texts(
            texts=b_texts,
            metadatas=b_metas,
            ids=b_ids,
        )

    # db.persist()
    print("✅ All documents upserted and persisted.")

def list_documents():
    print("🗂️ Listing documents in Chroma DB...")
    db = Chroma(
        persist_directory=CHROMADB_PATH,
        embedding_function=get_embedding_function(),
        collection_name=COLLECTION_NAME,
    )
    items = db.get()
    ids = items["ids"]
    print(f"📦 Total documents in DB: {len(ids)}")
    print("📝 First 40 IDs:", ids[:40])

def clear_database():
    if os.path.exists(CHROMADB_PATH):
        print("🧹 Deleting Chroma DB directory...")
        shutil.rmtree(CHROMADB_PATH)
        print("✅ Database cleared.")
    else:
        print("⚠️ No DB found at path to delete.")

if __name__ == "__main__":
    main()
