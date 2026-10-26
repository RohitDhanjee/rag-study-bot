import json
from pathlib import Path

import numpy as np

from embeddings import embed
from ingest import build_chunks

pdf = next(Path("data").glob("*.pdf"))
chunks = build_chunks(pdf)
print(f"Embedding {len(chunks)} chunks...")

vectors = embed([c["text"] for c in chunks], task_type="RETRIEVAL_DOCUMENT")
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # normalize every vector to length 1

Path("index").mkdir(exist_ok=True)
np.save("index/vectors.npy", vectors)
Path("index/chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

print(f"Index saved: {vectors.shape}")
