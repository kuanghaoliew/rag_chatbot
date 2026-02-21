"""
Retriever: Query ChromaDB for relevant chunks.
"""
import chromadb
from openai import OpenAI

from rag_chatbot.config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    COLLECTION_NAME, TOP_K, SIMILARITY_THRESHOLD,
)

client = OpenAI(api_key=OPENAI_API_KEY)


def get_collection():
    """Get ChromaDB collection."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return chroma_client.get_collection(name=COLLECTION_NAME)


def embed_query(query: str) -> list[float]:
    """Embed a single query."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a query.

    Returns list of dicts with keys: text, metadata, distance, similarity
    Only returns chunks with similarity >= SIMILARITY_THRESHOLD
    """
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]
        similarity = 1 - distance  # Convert distance to similarity score

        # Filter based on similarity threshold
        if similarity >= SIMILARITY_THRESHOLD:
            retrieved.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "similarity": similarity,
            })

    return retrieved


def format_context(retrieved_chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        meta = chunk["metadata"]
        source_info = f"[Source: {meta.get('type', 'unknown')}, Page: {meta.get('page', '?')}]"
        context_parts.append(f"--- Chunk {i+1} {source_info} ---\n{chunk['text']}")

    return "\n\n".join(context_parts)
