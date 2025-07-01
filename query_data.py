import argparse
import time
import os
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from constants import CHROMADB_PATH
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")

COLLECTION_NAME = "test"

PROMPT_TEMPLATE = """
You are an expert assistant. Using only the provided context, answer the user's question clearly and helpfully.

Follow these rules:
- DO NOT use outside knowledge.
- If the answer is not in the context, respond: "I don't know based on the provided context."
- If the context includes image references, show them using markdown: ![](path)
- Use friendly formatting: headings, bullet points, icons (✅, 💡, 📘) where relevant.
- Be conversational, but stay professional.
- End your response with a helpful suggestion or invitation for a follow-up question.

Context:
{context}

Question:
{question}

Answer:
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    embedding_function = OpenAIEmbeddings()
    db = Chroma(
        persist_directory=CHROMADB_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    query_rag(query_text, db)

def query_rag(query_text: str, db):
    results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        print("\n📘 Answer:\nI don't know based on the provided context.")
        print("\n🔍 Sources Used: none")
        return "I don't know based on the provided context."

    # Sort by similarity score
    final_results = sorted(results, key=lambda x: x[1])

    context_parts = []
    for doc, _ in final_results:
        text = doc.page_content.strip()
        image_path = doc.metadata.get("image_path")
        if image_path:
            text += f"\n\n![]({image_path})"
        context_parts.append(text)

    context_text = "\n\n---\n\n".join(context_parts)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    start_time = time.time()
    model = ChatOpenAI(model="gpt-3.5-turbo")
    response = model.invoke(prompt)
    response_text = response.content.strip()
    end_time = time.time()

    if len(response_text) < 300 and "I don't know" not in response_text:
        response_text += "\n\n💡 Would you like help exploring the diagrams or definitions further?"

    sources = [doc.metadata.get("id", "unknown") for doc, _ in final_results]
    print("\n📘 Answer:")
    print(response_text)
    print("\n🔍 Sources Used:")
    print(", ".join(sources))

    return response_text

if __name__ == "__main__":
    main()