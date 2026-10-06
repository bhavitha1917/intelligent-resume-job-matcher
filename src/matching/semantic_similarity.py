"""Model 3 (spec.md section 15): pretrained Sentence Transformer semantic
similarity. We do NOT fine-tune or train a transformer from scratch; the
model is downloaded/cached by the `sentence-transformers` library on first
use rather than being bundled into Git (see .gitignore).
"""

from functools import lru_cache
from typing import List

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_cache: dict[str, np.ndarray] = {}


@lru_cache(maxsize=1)
def _get_model(model_name: str = DEFAULT_MODEL_NAME):
    # Imported lazily so modules that only need keyword/TF-IDF matching
    # don't require torch/sentence-transformers to be installed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def encode_text(text: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Encode a single text into an embedding, cached by exact text content
    so repeated calls (e.g. re-ranking against the same job description)
    don't re-run the model.
    """
    if not text or not text.strip():
        model = _get_model(model_name)
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)

    cache_key = f"{model_name}::{hash(text)}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]

    model = _get_model(model_name)
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    _embedding_cache[cache_key] = embedding
    return embedding


def encode_resume(resume_text: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    return encode_text(resume_text, model_name)


def encode_job_description(job_text: str, model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    return encode_text(job_text, model_name)


def semantic_similarity(resume_text: str, job_text: str, model_name: str = DEFAULT_MODEL_NAME) -> float:
    """Cosine similarity between resume and job-description embeddings.
    Embeddings are pre-normalized, so cosine similarity reduces to a dot
    product. Returns 0.0 for empty input.
    """
    if not resume_text or not job_text or not resume_text.strip() or not job_text.strip():
        return 0.0
    resume_vec = encode_resume(resume_text, model_name)
    job_vec = encode_job_description(job_text, model_name)
    similarity = float(np.dot(resume_vec, job_vec))
    return round(similarity, 4)


def batch_encode(texts: List[str], model_name: str = DEFAULT_MODEL_NAME) -> np.ndarray:
    """Encode many texts at once (more efficient than repeated encode_text
    calls during batch ranking of many candidates)."""
    model = _get_model(model_name)
    non_empty = [t if t and t.strip() else " " for t in texts]
    return model.encode(non_empty, convert_to_numpy=True, normalize_embeddings=True)
