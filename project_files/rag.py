"""Tiny policy-RAG: markdown files -> chunks per '## ' section -> bge-small embeddings
-> cosine search."""
import os
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

POLICY_DIR = Path(__file__).resolve().parent / "policies"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 3
MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.45"))  # tune using DEBUG_RAG output

_embedder = SentenceTransformer(EMBED_MODEL, device="cpu")  # keep the GPU for the chat model


def _load_chunks():
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        title = path.stem.replace("_", " ").title()
        parts = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.M)
        for part in parts[1:]:  
            heading, _, body = part.partition("\n")
            chunks.append({"source": title, "heading": heading.strip(), "text": body.strip()})
    return chunks


_chunks = _load_chunks()
_vectors = (
    _embedder.encode(
        [f"{c['source']} - {c['heading']}\n{c['text']}" for c in _chunks],
        normalize_embeddings=True,
    )
    if _chunks else np.zeros((0, 384))
)


def search_policies(query: str, k: int = TOP_K):
    if not _chunks:
        return []
    q = _embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
    scores = _vectors @ q
    top = np.argsort(-scores)[:k]
    if os.getenv("DEBUG_RAG", "true").lower() == "true":
        print("RAG:", [(_chunks[i]["heading"], round(float(scores[i]), 3)) for i in top])
    return [{**_chunks[i], "score": float(scores[i])} for i in top if scores[i] >= MIN_SCORE]