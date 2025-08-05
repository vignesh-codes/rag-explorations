"""Simple query processing."""

import time
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ..utils.config import OPENAI_API_KEY, GENERATION_MODEL
from ..utils.exceptions import QueryError

PROMPT_TEMPLATE = """
You are a helpful AI assistant that answers questions based on the provided document context. 
Provide comprehensive, well-structured answers that directly address the user's question.

Guidelines:
- Answer in complete sentences and paragraphs, not bullet points
- Provide detailed explanations when possible
- Include specific examples or details from the context
- If the context contains multiple relevant pieces of information, synthesize them into a coherent response
- Use natural, conversational language
- If the answer isn't fully covered in the context, clearly state what information is missing
- Organize longer answers with clear structure (introduction, main points, conclusion)

Context from documents:
{context}

Question: {question}

Provide a comprehensive answer based on the context above:
"""

class QueryEngine:
    """Simple query engine."""
    
    def __init__(self):
        self.model = ChatOpenAI(
            model=GENERATION_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1
        )
        self.prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    def query(self, database, question, k=5):
        """Process a query with enhanced metrics."""
        start_time = time.time()
        
        # Search for relevant documents
        results = database.search(question, k=k)
        
        if not results:
            return {
                "answer": "I don't know based on the provided context.",
                "sources": [],
                "response_time": time.time() - start_time,
                "retrieved_docs": 0,
                "context_length": 0,
                "similarity_scores": []
            }
        
        # Prepare context
        context_parts = []
        sources = []
        similarity_scores = []
        
        for doc, score in results:
            context_parts.append(doc.page_content)
            sources.append(doc.metadata.get("source", "unknown"))
            similarity_scores.append(float(score))
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate response
        formatted_prompt = self.prompt.format(context=context, question=question)
        response = self.model.invoke(formatted_prompt)
        
        # Get document IDs and metadata for the retrieved documents
        retrieved_docs_info = []
        for doc, score in results:
            retrieved_docs_info.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'similarity_score': float(score)
            })
        
        return {
            "answer": response.content.strip(),
            "sources": sources,
            "response_time": time.time() - start_time,
            "retrieved_docs": len(results),
            "context_length": len(context),
            "prompt_length": len(formatted_prompt),
            "similarity_scores": similarity_scores,
            "avg_similarity": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
            "max_similarity": max(similarity_scores) if similarity_scores else 0,
            "min_similarity": min(similarity_scores) if similarity_scores else 0,
            "retrieved_docs_info": retrieved_docs_info
        }