import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap.umap_ as umap

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from constants import CHROMADB_PATH
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
COLLECTION_NAME = "test"

VISUALIZATION_PATH = "visualizations"

def visualize_chroma_chunks():
    print("📊 Generating visualizations...")

    db = Chroma(
        persist_directory=CHROMADB_PATH,
        embedding_function=OpenAIEmbeddings(),
        collection_name=COLLECTION_NAME,
    )

    items = db.get()
    ids = items["ids"]
    metadatas = items["metadatas"]
    documents = items["documents"]
    embeddings = items["embeddings"]

    records = []
    for i, meta in enumerate(metadatas):
        records.append({
            "id": meta.get("id"),
            "source": meta.get("source"),
            "page": int(meta.get("page", 0)),
            "chunk_index": int(meta.get("chunk_index", 0)),
            "text_length": len(documents[i].split()),
            "embedding": embeddings[i] if embeddings and embeddings[i] else np.zeros(1536)
        })

    df = pd.DataFrame(records)

    # Plot 1: Chunks per Page
    plt.figure(figsize=(10, 5))
    df.groupby("page").size().plot(kind='bar', color='slateblue')
    plt.title("Number of Chunks per Page")
    plt.xlabel("Page Number")
    plt.ylabel("Chunk Count")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_PATH}/chunks_per_page.png")

    # Plot 2: Chunk Size Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(df["text_length"], bins=20, color='skyblue', edgecolor='black')
    plt.title("Chunk Text Length Distribution")
    plt.xlabel("Word Count per Chunk")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_PATH}/chunk_size_distribution.png")

    # Plot 3: Chunk Density Heatmap by Page
    pivot = df.pivot_table(index="page", columns="chunk_index", values="text_length", fill_value=0)
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="Blues", cbar_kws={"label": "Words per Chunk"})
    plt.title("Heatmap: Chunk Density per Page")
    plt.xlabel("Chunk Index")
    plt.ylabel("Page Number")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_PATH}/heatmap_chunks.png")

    # Plot 4: Scatter Plot of Chunk Size by Page
    plt.figure(figsize=(10, 5))
    plt.scatter(df["page"], df["text_length"], alpha=0.6)
    plt.title("Chunk Size by Page")
    plt.xlabel("Page Number")
    plt.ylabel("Words in Chunk")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_PATH}/scatter_chunk_size_by_page.png")

    # Plot 5: Cumulative Chunk Growth
    df_sorted = df.sort_values(by=["page", "chunk_index"])
    df_sorted["cumulative_chunks"] = range(1, len(df_sorted) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(df_sorted["cumulative_chunks"], color="green")
    plt.xlabel("Chunk Number")
    plt.ylabel("Total Chunks Added")
    plt.title("Cumulative Chunk Growth Across Pages")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_PATH}/cumulative_chunk_growth.png")

    # Plot 6: UMAP Embedding Projection
    embeddings_matrix = np.array(df["embedding"].tolist())
    reducer = umap.UMAP(random_state=42)
    embedding_2d = reducer.fit_transform(embeddings_matrix)

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=df["page"],
        cmap="Spectral",
        s=30,
        alpha=0.8,
        edgecolor='k'
    )
    plt.colorbar(scatter, label="Page Number")
    plt.title("UMAP Projection of Chunk Embeddings (Colored by Page)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_PATH}/umap_chunk_projection.png")

    print("📊 Charts saved:")
    print("  - chunks_per_page.png")
    print("  - chunk_size_distribution.png")
    print("  - heatmap_chunks.png")
    print("  - scatter_chunk_size_by_page.png")
    print("  - cumulative_chunk_growth.png")
    print("  - umap_chunk_projection.png")

def visualize_query_results(results):
    print("📊 Visualizing query similarity scores...")

    scored_chunks = []
    for doc, score in results:
        meta = doc.metadata
        scored_chunks.append({
            "chunk_id": meta.get("id"),
            "page": int(meta.get("page", 0)),
            "chunk_index": int(meta.get("chunk_index", 0)),
            "score": score,
            "text_length": len(doc.page_content.split())
        })

    df = pd.DataFrame(scored_chunks)

    # Plot 1: Top-K Chunk Match Scores
    plt.figure(figsize=(10, 5))
    sns.barplot(x="chunk_index", y="score", hue="page", data=df, dodge=False)
    plt.title("Similarity Scores of Retrieved Chunks")
    plt.xlabel("Chunk Index")
    plt.ylabel("Distance (lower = more relevant)")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_PATH}/query_similarity_scores.png")

    # Plot 2: Distribution of Retrieved Chunk Sizes
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="page", y="text_length", data=df)
    plt.title("Retrieved Chunk Lengths by Page")
    plt.xlabel("Page")
    plt.ylabel("Words in Chunk")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_PATH}/query_chunk_lengths.png")

    # Plot 3: Scatter of Score vs Length
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x="text_length", y="score", hue="page", data=df)
    plt.title("Score vs Chunk Length")
    plt.xlabel("Words in Chunk")
    plt.ylabel("Distance Score")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_PATH}/query_score_vs_length.png")

    print("Query visualizations saved:")
    print("  - query_similarity_scores.png")
    print("  - query_chunk_lengths.png")
    print("  - query_score_vs_length.png")

def visualize_embedding_projection(all_embeddings, retrieved_embeddings, query_embedding, augmented_query_embedding=None):
    reducer = umap.UMAP(random_state=42)
    all_proj = reducer.fit_transform(np.array(all_embeddings))
    query_proj = reducer.transform(np.array(query_embedding))
    if augmented_query_embedding is not None:
        aug_proj = reducer.transform(np.array(augmented_query_embedding))
    retrieved_proj = reducer.transform(np.array(retrieved_embeddings))

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(all_proj[:, 0], all_proj[:, 1], s=15, color="gray", label="All Chunks")
    plt.scatter(retrieved_proj[:, 0], retrieved_proj[:, 1], s=80, edgecolors="green", facecolors="none", linewidths=2, label="Top-k Retrieved Chunks")
    plt.scatter(query_proj[:, 0], query_proj[:, 1], s=100, marker="X", color="red", label="Original Query")
    if augmented_query_embedding is not None:
        plt.scatter(aug_proj[:, 0], aug_proj[:, 1], s=100, marker="X", color="orange", label="Augmented Query")

    plt.title("UMAP Projection of Query and Retrieved Chunks")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATION_PATH}/query_umap_projection.png")
    print("Saved query_umap_projection.png")

if __name__ == "__main__":
    visualize_chroma_chunks()
