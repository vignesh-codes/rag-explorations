"""Simple query processing."""

import time
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from ..utils.config import OPENAI_API_KEY, GENERATION_MODEL
from ..utils.exceptions import QueryError

PROMPT_TEMPLATE = """
Using only the provided context, answer the user's question clearly and helpfully.

Rules:
- Use only information from the context
- If the answer isn't in the context, say "I don't know based on the provided context"
- Structure your response with bullet points where helpful
- Be conversational but professional

Context:
{context}

Question: {question}

Answer:
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
        """Process a query."""
        start_time = time.time()
        
        # Search for relevant documents
        results = database.search(question, k=k)
        
        if not results:
            return {
                "answer": "I don't know based on the provided context.",
                "sources": [],
                "response_time": time.time() - start_time
            }
        
        # Prepare context
        context_parts = []
        sources = []
        
        for doc, score in results:
            context_parts.append(doc.page_content)
            sources.append(doc.metadata.get("source", "unknown"))
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate response
        prompt = self.prompt.format(context=context, question=question)
        response = self.model.invoke(prompt)
        
        return {
            "answer": response.content.strip(),
            "sources": sources,
            "response_time": time.time() - start_time
        }