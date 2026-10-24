import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from embeddings import embed

vectors = np.load("index/vectors.npy")
chunks = json.loads(Path("index/chunks.json").read_text(encoding="utf-8"))


def tokenize(text):
    """Text -> chhote lafzon ki list. 'load-store' jaise lafz ek hi token rehte hain."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])


def semantic_search(query, n):
    """Matlab se dhoondo: embeddings + cosine. Chunk index ki ranked list wapas do."""
    q = embed([query], task_type="RETRIEVAL_QUERY")[0]
    q = q / np.linalg.norm(q)
    scores = vectors @ q
    return list(np.argsort(scores)[::-1][:n])


def keyword_search(query, n):
    """Lafzon se dhoondo: BM25. Chunk index ki ranked list wapas do."""
    scores = bm25.get_scores(tokenize(query))
    return list(np.argsort(scores)[::-1][:n])


def search(query, k=3, n=10, rrf_k=60):
    """Hybrid search: dono rankings ko Reciprocal Rank Fusion se jodo."""
    fused = {}
    for ranking in (semantic_search(query, n), keyword_search(query, n)):
        for rank, i in enumerate(ranking):
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + rank + 1)
    top = sorted(fused, key=fused.get, reverse=True)[:k]
    return [(fused[i], chunks[i]) for i in top]


if __name__ == "__main__":
    while True:
        query = input("\nSawal poochiye (khali Enter = exit): ").strip()
        if not query:
            break
        for score, c in search(query):
            print(f"\n[{score:.4f}] page {c['page']}")
            print("   " + c["text"][:200] + "...")