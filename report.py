"""
Legal document report: RAG-free version using chunk summarization + predefined queries.
"""

from __future__ import annotations

import json
import os
import re

import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PREDEFINED_QUERIES = {
    "termination": "What are the termination conditions or exit clauses?",
    "liability": "What liability limitations or risk allocations exist?",
    "payment": "What are the payment terms, fees, and billing structure?",
    "confidentiality": "What confidentiality or NDA clauses are included?",
    "risks": "What unusual, risky, or one-sided clauses exist?",
}

QUERY_KEYS = tuple(PREDEFINED_QUERIES.keys())


# ---------
# Chunking
# ---------
def chunk_text(text: str, max_chars: int = 2800, overlap: int = 80) -> list[str]:
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


# ------------
# LLM wrapper 
# ------------
class LLM:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str, *, max_output_tokens: int = 900) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": max_output_tokens,
            },
        )
        return (response.text or "").strip()


# ----------------------------
# Step 1: chunk summarization
# ----------------------------
def _summarize_chunk(llm: LLM, chunk: str) -> str:
    prompt = (
        "Summarize legal facts only (3–6 bullets). Ignore filler. Be precise.\n\n"
        f"{chunk}"
    )
    return llm.generate(prompt, max_output_tokens=700)


def _build_compressed_document(llm: LLM, chunks: list[str]) -> str:
    """
    Map step: summarize chunks
    Reduce step: merge into a compact document
    """
    summaries = [_summarize_chunk(llm, c) for c in chunks]
    if len(summaries) == 1:
        return summaries[0]

    merge_prompt = (
        "Merge these chunk summaries into one compact outline. Dedupe; keep legal specifics.\n\n"
        + "\n---\n".join(summaries)
    )
    return llm.generate(merge_prompt, max_output_tokens=1200)


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t, count=1)
    return t.strip()


def _json_object_substring(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]


def _parse_analysis_json(raw: str) -> dict[str, str]:
    """Parse batched JSON; tolerate fences and leading prose."""
    cleaned = _strip_code_fence(raw)
    candidates = [cleaned]
    inner = _json_object_substring(cleaned)
    if inner and inner not in candidates:
        candidates.append(inner)

    data: object | None = None
    for candidate in candidates:
        if not candidate.strip():
            continue
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    out: dict[str, str] = {}
    if not isinstance(data, dict):
        return out
    for key in QUERY_KEYS:
        val = data.get(key)
        out[key] = (val if isinstance(val, str) else "").strip()
    return out


def _batched_analysis_prompt(compressed_doc: str) -> str:
    lines = [f'- "{k}": {q}' for k, q in PREDEFINED_QUERIES.items()]
    q_block = "\n".join(lines)
    return (
        "You analyze a legal document excerpt. Use ONLY the text in <doc>. "
        'If missing, use exactly: Not found in document\n'
        "Rules: no speculation; concise bullets.\n\n"
        f"<doc>\n{compressed_doc}\n</doc>\n\n"
        "Return ONLY valid JSON with these string keys and your answers:\n"
        f"{q_block}"
    )


def analyze_document(llm: LLM | None = None, compressed_doc: str = "") -> dict[str, str]:
    """
    Runs fixed queries over compressed document (NO RAG) in one model call.
    """
    client = llm or LLM()
    prompt = _batched_analysis_prompt(compressed_doc)
    raw = client.generate(prompt, max_output_tokens=2048)
    parsed = _parse_analysis_json(raw)

    def _all_answered(d: dict[str, str]) -> bool:
        return all(d.get(k) for k in QUERY_KEYS)

    if _all_answered(parsed):
        return parsed

    retry_prompt = (
        _batched_analysis_prompt(compressed_doc)
        + "\n\nOutput: one JSON object only. No markdown fences or commentary."
    )
    raw2 = client.generate(retry_prompt, max_output_tokens=2048)
    parsed2 = _parse_analysis_json(raw2)

    merged: dict[str, str] = {}
    for k in QUERY_KEYS:
        v = (parsed.get(k) or parsed2.get(k) or "").strip()
        merged[k] = v if v else "Not found in document"
    return merged


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
