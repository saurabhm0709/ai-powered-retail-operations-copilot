"""Chunks knowledge_base/*.md by "## " section and stores them in ChromaDB (.chroma/). Idempotent.

Usage: python scripts/build_index.py [--rebuild]
Embeddings use ChromaDB's default local model (all-MiniLM-L6-v2): free, no API key.
"""

import argparse
import sys
from pathlib import Path

# Lets "python scripts/build_index.py" import the copilot package from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402

from copilot import config, rag  # noqa: E402


# Deletes and recreates the collection, then embeds and adds every chunk.
def build_collection(client: chromadb.ClientAPI, chunks: list[dict]) -> chromadb.Collection:
    existing_names = [collection.name for collection in client.list_collections()]
    if config.CHROMA_COLLECTION in existing_names:
        client.delete_collection(config.CHROMA_COLLECTION)
    # Cosine distance so RAG_MAX_DISTANCE means the same thing for short and long sections.
    collection = client.create_collection(config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(ids=[chunk["id"] for chunk in chunks],
                   documents=[chunk["text"] for chunk in chunks],
                   metadatas=[chunk["metadata"] for chunk in chunks])
    return collection


# True when the saved collection already holds exactly these chunk ids (nothing to do).
def index_is_current(client: chromadb.ClientAPI, chunks: list[dict]) -> bool:
    existing_names = [collection.name for collection in client.list_collections()]
    if config.CHROMA_COLLECTION not in existing_names:
        return False
    saved_ids = client.get_collection(config.CHROMA_COLLECTION).get()["ids"]
    return sorted(saved_ids) == sorted(chunk["id"] for chunk in chunks)


# Builds the index if missing, changed, or --rebuild; prints the chunk ids either way.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="re-embed even if the index is current")
    args = parser.parse_args()
    chunks = rag.chunk_knowledge_base()
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    if not args.rebuild and index_is_current(client, chunks):
        print(f"Index is current: {len(chunks)} chunks (use --rebuild to force).")
    else:
        build_collection(client, chunks)
        print(f"Built index: {len(chunks)} chunks in {config.CHROMA_DIR}")
    for chunk in chunks:
        print(" ", chunk["id"])


if __name__ == "__main__":
    main()
