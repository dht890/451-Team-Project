"""
Legal document report: RAG-free version using chunk summarization + predefined queries.
No FAISS, no embeddings, no retrieval.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from openai import OpenAI

PREDEFINED_QUERIES = {
    "termination": "What are the termination conditions or exit clauses?",
    "liability": "What liability limitations or risk allocations exist?",
    "payment": "What are the payment terms, fees, and billing structure?",
    "confidentiality": "What confidentiality or NDA clauses are included?",
    "risks": "What unusual, risky, or one-sided clauses exist?",
}


# ----------------------------
# Chunking (unchanged concept)
# ----------------------------
def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
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


# ----------------------------
# LLM wrapper (unchanged API)
# ----------------------------
class LLM:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI()

    def generate(self, prompt: str) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=900,
        )
        return (completion.choices[0].message.content or "").strip()


# ----------------------------
# Step 1: chunk summarization
# ----------------------------
def _summarize_chunk(llm: LLM, chunk: str) -> str:
    prompt = f"""
Summarize the following legal text chunk.

Rules:
- Keep only key legal facts
- Ignore filler text
- Use 3–6 bullet points max
- Be precise

Text:
{chunk}
"""
    return llm.generate(prompt)


def _build_compressed_document(llm: LLM, chunks: list[str]) -> str:
    """
    Map step: summarize chunks
    Reduce step: merge into a compact document
    """
    summaries = [_summarize_chunk(llm, c) for c in chunks]

    merge_prompt = f"""
Combine the following chunk summaries into a single structured legal document.

Rules:
- Remove duplicates
- Keep important legal clauses
- Organize clearly by topics if possible

Chunk summaries:
{chr(10).join(summaries)}
"""
    return llm.generate(merge_prompt)


# ----------------------------
# Step 2: predefined queries
# ----------------------------
def _analysis_prompt(question: str, document: str) -> str:
    return f"""
You are analyzing a legal document.

Answer using ONLY the document below.
If not found, respond exactly: Not found in document

Rules:
- No speculation
- Bullet points only
- Be concise

Document:
{document}

Question:
{question}
"""


def analyze_document(llm: LLM | None = None, compressed_doc: str = "") -> dict[str, str]:
    """
    Runs fixed queries over compressed document (NO RAG).
    """
    client = llm or LLM()
    results: dict[str, str] = {}

    for key, question in PREDEFINED_QUERIES.items():
        prompt = _analysis_prompt(question, compressed_doc)
        results[key] = client.generate(prompt)

    return results


# ----------------------------
# Main pipeline (RAG-free)
# ----------------------------
def run_report_on_text(full_text: str, llm: LLM | None = None) -> dict[str, str]:
    """
    Full pipeline:
    1. chunk
    2. summarize chunks
    3. compress document
    4. run predefined queries
    """
    client = llm or LLM()

    chunks = chunk_text(full_text)
    if not chunks:
        return {k: "No extractable text in the document." for k in PREDEFINED_QUERIES}

    compressed_doc = _build_compressed_document(client, chunks)

    return analyze_document(llm=client, compressed_doc=compressed_doc)
