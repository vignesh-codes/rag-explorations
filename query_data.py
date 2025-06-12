import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM as Ollama
from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
COLLECTION_NAME = "test"

PROMPT_TEMPLATE = """
You are an expert question answering system. Answer the question based only on the provided context below.

If the answer is not contained in the context, say "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function,
        collection_name=COLLECTION_NAME
    )

    # Run RAG query
    query_rag(query_text, db)

def query_rag(query_text: str, db):
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    # Optional filtering based on query keywords.
    # Example: if question is about "monopoly", prefer monopoly chunks.
    apply_filter = "monopoly" in query_text.lower()
    filtered_results = []

    if apply_filter:
        for doc, score in results:
            if "monopoly" in doc.metadata.get("source", "").lower():
                filtered_results.append((doc, score))

    # Use filtered results if available, otherwise use full results.
    final_results = filtered_results if filtered_results else results

    # Optional: sort final results by score (lower distance = more similar).
    final_results = sorted(final_results, key=lambda x: x[1])

    # If no chunks retrieved → avoid sending empty context.
    if not final_results:
        print("⚠️ No relevant context found in database.")
        print("Response: I don't know based on the provided context.")
        return "I don't know based on the provided context."

    # Build prompt with retrieved chunks.
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in final_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Call LLM (Ollama mistral).
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    # Print formatted response with sources.
    sources = [doc.metadata.get("id", None) for doc, _score in final_results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

    return response_text

if __name__ == "__main__":
    main()
