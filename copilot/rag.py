"""[3A] RAG Agent: chunks the KB by "## " section (used by build_index) and retrieves from the saved index."""

import re
from functools import lru_cache
from pathlib import Path

import chromadb

from copilot import config
from copilot.schemas import PolicyHit


# Section heading -> citation slug ("POS Card Reader Runbook" -> "pos-card-reader-runbook").
def slugify(heading: str) -> str:
    slug = re.sub(r"[^a-z0-9 -]", "", heading.lower())
    return slug.strip().replace(" ", "-")


# KB file text -> "Source ID:" value from the file header, e.g. "INV-REPLEN-03".
def read_source_id(file_text: str) -> str:
    match = re.search(r"^Source ID:\s*(\S+)", file_text, re.MULTILINE)
    return match.group(1) if match else ""


# KB file text -> (doc title, list of (section heading, section body)). Text before the first "## " is header only.
def split_sections(file_text: str) -> tuple[str, list[tuple[str, str]]]:
    title_match = re.search(r"^# (.+)$", file_text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else ""
    parts = re.split(r"^## (.+)$", file_text, flags=re.MULTILINE)
    # re.split with a capture group gives [header, heading1, body1, heading2, body2, ...]
    sections = []
    for index in range(1, len(parts), 2):
        sections.append((parts[index].strip(), parts[index + 1].strip()))
    return title, sections


# One KB file -> list of chunk dicts {id, text, metadata}; chunk id == citation.
def chunk_file(path: Path) -> list[dict]:
    file_text = path.read_text(encoding="utf-8")
    title, sections = split_sections(file_text)
    source_id = read_source_id(file_text)
    chunks = []
    for heading, body in sections:
        slug = slugify(heading)
        chunks.append({
            "id": f"{path.name}#{slug}",
            "text": f"{title} — {heading}\n{body}",
            "metadata": {"file": path.name, "section": heading, "slug": slug, "source_id": source_id},
        })
    return chunks


# All knowledge_base/*.md files -> list of chunks, in file order.
def chunk_knowledge_base() -> list[dict]:
    chunks = []
    for path in sorted(config.KB_DIR.glob("*.md")):
        chunks.extend(chunk_file(path))
    return chunks


# Opens the saved ChromaDB collection built by scripts/build_index.py (never re-embeds documents).
@lru_cache(maxsize=1)
def open_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.CHROMA_COLLECTION)


# Chunk dict + score -> PolicyHit.
def to_policy_hit(chunk: dict, score: float) -> PolicyHit:
    metadata = chunk["metadata"]
    return PolicyHit(citation=chunk["id"], file=metadata["file"], section=metadata["section"],
                     source_id=metadata["source_id"], text=chunk["text"], score=round(score, 3))


# Lowercase words of 3+ letters, used by the mock keyword-overlap retriever.
def word_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


# MOCK_LLM retrieval: rank chunks by share of query words they contain (no model needed).
def retrieve_by_keywords(query_text: str, k: int) -> list[PolicyHit]:
    query_words = word_set(query_text)
    scored = []
    for chunk in chunk_knowledge_base():
        overlap = len(query_words & word_set(chunk["text"]))
        if overlap:
            scored.append((overlap / len(query_words), chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [to_policy_hit(chunk, score) for score, chunk in scored[:k]]


# Real retrieval: local embedding search; keeps hits within RAG_MAX_DISTANCE. Score = 1 - distance.
def retrieve_by_embedding(query_text: str, k: int) -> list[PolicyHit]:
    result = open_collection().query(query_texts=[query_text], n_results=k)
    hits = []
    for chunk_id, text, metadata, distance in zip(result["ids"][0], result["documents"][0],
                                                  result["metadatas"][0], result["distances"][0]):
        if distance <= config.RAG_MAX_DISTANCE:
            hits.append(to_policy_hit({"id": chunk_id, "text": text, "metadata": metadata}, 1 - distance))
    return hits


# Masked question + category -> policy hits ([] means policy cannot be confirmed).
def retrieve(masked_query: str, category: str, k: int = config.RAG_TOP_K) -> list[PolicyHit]:
    query_text = f"{masked_query} | category: {category}"
    if config.MOCK_LLM:
        return retrieve_by_keywords(query_text, k)
    return retrieve_by_embedding(query_text, k)


# Citation id -> full section text (for the UI and the responder), or "" if unknown.
def get_section(citation: str) -> str:
    for chunk in chunk_knowledge_base():
        if chunk["id"] == citation:
            return chunk["text"]
    return ""
