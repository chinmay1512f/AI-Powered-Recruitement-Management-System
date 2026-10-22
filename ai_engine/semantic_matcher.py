"""Semantic similarity between a job description and a resume.

all-MiniLM-L6-v2 accepts at most 256 word-piece tokens and SILENTLY TRUNCATES
anything longer. A two-page resume is 400-1000 tokens, so the previous
whole-document encode only ever saw roughly its first quarter - and the score
changed by ~20 points depending on whether the candidate happened to put their
experience above or below their hobbies.

This module splits the resume into overlapping windows that each fit inside the
limit, embeds every window, and aggregates. Every part of the resume is now
read, and section order no longer decides the score.
"""
import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# This logger emits "Token indices sequence length is longer than..." whenever we
# tokenise a long resume to measure it. That is expected here - we are measuring
# precisely so we can split below the limit - so silence that one logger only.
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

MAX_TOKENS = int(getattr(model, "max_seq_length", 256) or 256)
CHUNK_TOKENS = MAX_TOKENS - 16   # headroom for [CLS] / [SEP]
OVERLAP_TOKENS = 40              # so a sentence split across a boundary is not lost

# How much the single best-matching section counts versus the average of the
# best few. Pure max lets one lucky line carry an irrelevant resume; pure mean
# punishes a strong specialist whose relevant work is one section of many.
BEST_CHUNK_WEIGHT = 0.6
TOP_K = 3


def chunk_text(text, chunk_tokens=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    """Split text into overlapping windows that each fit the model's limit."""
    tokenizer = model.tokenizer
    ids = tokenizer.encode(text, add_special_tokens=False)

    if len(ids) <= chunk_tokens:
        return [text]

    chunks = []
    step = max(chunk_tokens - overlap, 1)
    start = 0
    while start < len(ids):
        window = ids[start:start + chunk_tokens]
        if not window:
            break
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        if start + chunk_tokens >= len(ids):
            break
        start += step
    return chunks


def match_resume(job_text, resume_text, top_k=TOP_K):
    """Return a 0-100 similarity between a job description and a resume."""
    if not job_text.strip() or not resume_text.strip():
        return 0.0

    chunks = chunk_text(resume_text)
    embeddings = model.encode([job_text] + chunks)

    similarities = cosine_similarity(embeddings[0:1], embeddings[1:])[0]
    similarities = np.clip(similarities, 0.0, 1.0)

    best_first = np.sort(similarities)[::-1][:max(1, top_k)]
    score = (
        BEST_CHUNK_WEIGHT * float(best_first[0])
        + (1.0 - BEST_CHUNK_WEIGHT) * float(best_first.mean())
    )
    return round(score * 100, 2)


def match_resume_detailed(job_text, resume_text, top_k=TOP_K):
    """Same as match_resume, plus per-chunk scores. Useful for debugging."""
    if not job_text.strip() or not resume_text.strip():
        return 0.0, []

    chunks = chunk_text(resume_text)
    embeddings = model.encode([job_text] + chunks)
    similarities = np.clip(cosine_similarity(embeddings[0:1], embeddings[1:])[0], 0.0, 1.0)

    detail = [
        {"chunk": i, "score": round(float(s) * 100, 2), "preview": c[:80]}
        for i, (s, c) in enumerate(zip(similarities, chunks))
    ]
    return match_resume(job_text, resume_text, top_k), detail
