"""Minimal legal-style report: chunk retrieval + predefined Q&A via LLM."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable

from openai import OpenAI

PREDEFINED_QUERIES = {
    "termination": "What are the termination conditions or exit clauses?",
    "liability": "What liability limitations or risk allocations exist?",
    "payment": "What are the payment terms, fees, and billing structure?",
    "confidentiality": "What confidentiality or NDA clauses are included?",
    "risks": "What unusual, risky, or one-sided clauses exist?",
}

# Set by run_report_on_text() before analyze_document() for retrieve_context(question).
_ACTIVE_CHUNKS: list[str] = []


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1}


def _score_chunk(question: str, chunk: str) -> float:
    q_tokens = _tokenize(question)
    c_tokens = _tokenize(chunk)
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = len(q_tokens & c_tokens)
    return overlap / (len(q_tokens) ** 0.5)


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[str]:
    """Split into overlapping windows for retrieval."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]
    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + max_chars, n)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def retrieve_context(question: str, top_k: int = 4) -> list[str]:
    """Return the best-matching chunks for the question (uses _ACTIVE_CHUNKS)."""
    if not _ACTIVE_CHUNKS:
        return []
    ranked = sorted(_ACTIVE_CHUNKS, key=lambda c: _score_chunk(question, c), reverse=True)
    return ranked[:top_k]


def _set_active_chunks(chunks: Iterable[str]) -> None:
    global _ACTIVE_CHUNKS
    _ACTIVE_CHUNKS = list(chunks)


class LLM:
    """Thin wrapper around OpenAI chat completions."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI()

    def generate(self, prompt: str) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        choice = completion.choices[0].message.content
        return (choice or "").strip()


def analyze_document(llm: LLM | None = None) -> dict[str, str]:
    results: dict[str, str] = {}
    client = llm or LLM()

    for key, question in PREDEFINED_QUERIES.items():
        chunks = retrieve_context(question)
        context = "\n\n".join(chunks)

        prompt = f"""
You are analyzing a legal document.

Answer the question using ONLY the context.

Question:
{question}

Context:
{context}

Return a concise answer.
""".strip()

        results[key] = client.generate(prompt)

    return results


def run_report_on_text(full_text: str, llm: LLM | None = None) -> dict[str, str]:
    """
    Chunk `full_text`, then run analyze_document() with predefined queries.
    Returns empty-answer placeholders if there is no text.
    """
    _set_active_chunks(chunk_text(full_text))
    if not _ACTIVE_CHUNKS:
        return {k: "No extractable text in the document." for k in PREDEFINED_QUERIES}
    return analyze_document(llm=llm)
