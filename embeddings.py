import os
from pathlib import Path
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--intfloat--multilingual-e5-small"
os.environ.setdefault("HF_HUB_OFFLINE", "1" if MODEL_CACHE.exists() else "0")   # go offline only if the model is already cached
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "intfloat/multilingual-e5-small"
model = SentenceTransformer(EMBED_MODEL)      # ~470 MB download on first run, then loaded from cache

# E5 needs to be told whether a text is a query or a document
PREFIXES = {
    "RETRIEVAL_QUERY": "query: ",
    "RETRIEVAL_DOCUMENT": "passage: ",
    "SEMANTIC_SIMILARITY": "query: ",
}


def embed(texts, task_type):
    """List of texts -> matrix of vectors (one row per text)."""
    prefix = PREFIXES[task_type]
    vectors = model.encode([prefix + t for t in texts], batch_size=16)
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(a, b):
    """Cosine of the angle between two vectors: 1 = same direction, 0 = unrelated."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
