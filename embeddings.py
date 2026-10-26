import os
from pathlib import Path
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--intfloat--multilingual-e5-small"
os.environ.setdefault("HF_HUB_OFFLINE", "1" if MODEL_CACHE.exists() else "0")   # model pehle se ho tabhi offline   # laptop par offline; server apni setting khud de sakta hai      # model download ho chuka hai, ab internet se check mat karo
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "intfloat/multilingual-e5-small"
model = SentenceTransformer(EMBED_MODEL)      # pehli dafa ~470 MB download hoga, phir cache se

# E5 model ko batana parta hai ke text sawal hai ya document (Gemini ke task_type jaisa)
PREFIXES = {
    "RETRIEVAL_QUERY": "query: ",
    "RETRIEVAL_DOCUMENT": "passage: ",
    "SEMANTIC_SIMILARITY": "query: ",
}


def embed(texts, task_type):
    """Texts ki list -> vectors ka matrix (har row ek text ka vector)."""
    prefix = PREFIXES[task_type]
    vectors = model.encode([prefix + t for t in texts], batch_size=16)
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(a, b):
    """Do vectors ke beech angle ka cosine: 1 = same direction, 0 = koi taalluq nahi."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))