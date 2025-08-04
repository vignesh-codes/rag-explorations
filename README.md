# Simple RAG System

A clean, simple Retrieval-Augmented Generation (RAG) system for document Q&A.

## Features

- **Document Processing**: Load PDFs and split into chunks
- **Vector Search**: Find relevant document chunks using embeddings
- **Question Answering**: Generate answers using retrieved context
- **Observability**: Integrated with Langtrace for monitoring

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` file with your API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```

3. Add your PDF documents to the `data/` folder

## Usage

**Setup Database:**
```bash
python cli.py db reset      # Clear database
python cli.py db populate   # Load documents
python cli.py db list       # Check documents
```

**Ask Questions:**
```bash
python cli.py query "What is this about?"
python cli.py query "How does this work?" --top-k 10
```

## Project Structure

```
src/
├── models/
│   ├── database.py          # Vector database operations
│   ├── document_processor.py # PDF loading and chunking
│   └── query_engine.py      # Question answering
└── utils/
    ├── config.py            # Configuration
    ├── embeddings.py        # Embedding providers
    ├── exceptions.py        # Error handling
    └── logging_config.py    # Logging setup
```

## How It Works

1. **Load Documents**: PDFs are loaded from `data/` folder
2. **Create Chunks**: Documents are split into overlapping chunks
3. **Generate Embeddings**: Each chunk gets an embedding vector
4. **Store in Database**: Chunks and embeddings stored in ChromaDB
5. **Query Processing**: User questions are embedded and matched against chunks
6. **Generate Answer**: Retrieved chunks provide context for LLM to generate answer

Simple and effective! Perfect for learning RAG concepts.