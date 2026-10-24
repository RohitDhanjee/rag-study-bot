import json
from pathlib import Path

import numpy as np

from embeddings import embed
from ingest import build_chunks

pdf = next(Path("data").glob("*.pdf"))
chunks = build_chunks(pdf)
print(f"{len(chunks)} chunks ke embeddings ban rahe hain...")

vectors = embed([c["text"] for c in chunks], task_type="RETRIEVAL_DOCUMENT")
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # har vector ki length 1 karo

Path("index").mkdir(exist_ok=True)
np.save("index/vectors.npy", vectors)
Path("index/chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

print(f"Index save ho gaya: {vectors.shape}")