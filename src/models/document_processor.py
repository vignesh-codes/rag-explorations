"""Simple document processing."""

import nltk
from langchain.schema.document import Document
from nltk.tokenize import sent_tokenize
from ..utils.config import MAX_CHUNK_TOKENS, MIN_CHUNK_TOKENS, OVERLAP_SENTENCES

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)

def count_tokens(text):
    """Count tokens (simplified as word count)."""
    return len(text.split())

def process_documents(documents):
    """Split documents into chunks."""
    chunks = []
    
    for doc in documents:
        sentences = sent_tokenize(doc.page_content)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= MAX_CHUNK_TOKENS:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            else:
                if current_tokens >= MIN_CHUNK_TOKENS:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    chunks.append(Document(
                        page_content=chunk_text,
                        metadata=doc.metadata.copy()
                    ))
                    
                    # Start new chunk with overlap
                    current_chunk = current_chunk[-OVERLAP_SENTENCES:]
                    current_tokens = sum(count_tokens(s) for s in current_chunk)
                
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy()
            ))
    
    return chunks