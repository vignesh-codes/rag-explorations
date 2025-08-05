# PDF Chat Assistant with Performance Analytics

A Streamlit web application that allows users to upload PDF documents, chat with them using AI, and view detailed performance metrics about the RAG system.

## Features

### 📚 PDF Processing
- Upload multiple PDF files at once
- Automatic text extraction and chunking
- Vector database indexing for fast retrieval

### 💬 Interactive Chat
- Natural language queries about your PDFs
- Context-aware responses with source citations
- Chat history with performance metrics for each query

### 📊 Performance Analytics
- **Response Time**: How fast the system responds
- **Confidence Score**: Estimated confidence in the answer
- **Source Diversity**: How varied the retrieved sources are
- **Similarity Scores**: How well documents match the query
- **Context Length**: Amount of text used for generation
- **Query Complexity**: Analysis of question difficulty

### 📈 Analytics Dashboard
- Performance trends over time
- Summary statistics across all queries
- Visual charts showing system performance

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### 2. Set Up Environment
Make sure your `.env` file contains:
```
OPENAI_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_MODEL=gpt-4o-mini
```

### 3. Run the App
```bash
python run_streamlit.py
```

Or directly with Streamlit:
```bash
streamlit run streamlit_app.py
```

### 4. Use the App
1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
2. **Process Files**: Click "Process PDFs" to index them in the vector database
3. **Start Chatting**: Ask questions about your uploaded documents
4. **View Analytics**: Check the Analytics tab for performance insights

## Performance Metrics Explained

### Response Time
- **What it measures**: Total time from query to response
- **Good range**: < 3 seconds
- **Factors**: Document retrieval speed, LLM generation time

### Confidence Score
- **What it measures**: Estimated reliability of the answer
- **Range**: 0-100%
- **Factors**: Number of sources, answer length, uncertainty phrases

### Source Diversity
- **What it measures**: How varied the retrieved sources are
- **Range**: 0-100%
- **Higher is better**: More diverse sources = more comprehensive answers

### Similarity Scores
- **What it measures**: How well documents match your query
- **Range**: 0-1 (higher is better)
- **Use case**: Identify if relevant documents were found

## Use Cases

### 📖 Academic Research
- Upload research papers and textbooks
- Ask questions about specific concepts
- Track understanding through confidence scores

### 📋 Document Analysis
- Upload contracts, reports, manuals
- Extract key information quickly
- Monitor system performance for different document types

### 🎓 Learning Management
- Upload course materials
- Students can ask questions 24/7
- Professors can see what students are confused about

## Technical Architecture

```
PDF Upload → Text Extraction → Chunking → Vector Embedding → Vector DB
                                                                    ↓
User Query → Query Embedding → Similarity Search → Context → LLM → Response
                                                                    ↓
                                                            Performance Metrics
```

## Performance Optimization Tips

### For Better Response Times
- Use smaller, more focused PDFs
- Reduce the number of retrieved documents (k parameter)
- Use faster embedding models

### For Higher Confidence
- Upload high-quality, relevant documents
- Ask specific, clear questions
- Ensure documents contain the information you're seeking

### For Better Source Diversity
- Upload documents from different sources/authors
- Ask broader questions that require multiple perspectives

## Troubleshooting

### Common Issues

**"No documents found"**
- Check if PDFs contain extractable text (not just images)
- Ensure PDFs are not password-protected

**"Low confidence scores"**
- Upload more relevant documents
- Ask more specific questions
- Check if your question matches the document content

**"Slow response times"**
- Reduce the number of documents
- Use a faster OpenAI model
- Check your internet connection

## Future Enhancements

- [ ] Support for more file formats (Word, PowerPoint, etc.)
- [ ] Advanced analytics with charts and graphs
- [ ] User authentication and document management
- [ ] API endpoints for integration
- [ ] Batch processing capabilities
- [ ] Custom embedding models
- [ ] Multi-language support

## Contributing

This is part of a larger RAG system. To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the Streamlit app
5. Submit a pull request

## License

[Your License Here]