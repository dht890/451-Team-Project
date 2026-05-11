#report.py
"""
Legal document report: chunk extraction → Python merge → one batched JSON query.
Optimized for low token use: small chunks, minimal overlap, compact prompts, caching.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict

import google.genai as genai
from dotenv import load_dotenv

import pandas as pd

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PREDEFINED_QUERIES = {
    "termination": "What are the termination conditions or exit clauses?",
    "liability": "What liability limitations or risk allocations exist?",
    "payment": "What are the payment terms, fees, and billing structure?",
    "confidentiality": "What confidentiality or NDA clauses are included?",
    "risks": "What unusual, risky, or one-sided clauses exist?",
    "pricing_risk": "What clauses may allow price increases, auto-renewals, or unclear billing changes?",
}

QUERY_KEYS = tuple(PREDEFINED_QUERIES.keys())

_OPTIONAL_REPORT_KEYS = ("price_anomalies", "price_anomalies_warning")


def _coerce_analysis_value(val: object) -> str:
    """Turn model JSON values into plain text for each report section."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, int | float):
        return str(val)
    if isinstance(val, list | tuple):
        parts = [_coerce_analysis_value(x) for x in val]
        return "\n".join(p for p in parts if p)
    if isinstance(val, dict):
        try:
            return json.dumps(val, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(val)
    return str(val).strip()


def _finalize_report_payload(src: dict[str, object]) -> dict[str, str]:
    """Always emit QUERY_KEYS plus optional pricing keys (stable API + UI)."""
    out: dict[str, str] = {}
    for k in QUERY_KEYS:
        s = _coerce_analysis_value(src.get(k)).strip()
        out[k] = s if s else "Not found in document"
    for k in _OPTIONAL_REPORT_KEYS:
        if k not in src:
            continue
        s = _coerce_analysis_value(src.get(k)).strip()
        if s:
            out[k] = s
    return out


# LRU caps to avoid unbounded memory on long sessions
_CHUNK_CACHE_MAX = 2000
_REPORT_CACHE_MAX = 64


def clean_text(text: str) -> str:
    """Collapse whitespace/newlines — fewer tokens, same semantics."""
    return " ".join(text.split())


def _sha256_utf8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class _LRUCache:
    def __init__(self, max_items: int) -> None:
        self._max = max_items
        self._data: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        val = self._data.get(key)
        if val is not None:
            self._data.move_to_end(key)
        return val

    def set(self, key: str, value: str) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._max:
            self._data.popitem(last=False)


class _ReportLRUCache:
    def __init__(self, max_items: int) -> None:
        self._max = max_items
        self._data: OrderedDict[str, dict[str, str]] = OrderedDict()

    def get(self, key: str) -> dict[str, str] | None:
        val = self._data.get(key)
        if val is not None:
            self._data.move_to_end(key)
        return val

    def set(self, key: str, value: dict[str, str]) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self._max:
            self._data.popitem(last=False)


_chunk_extraction_cache = _LRUCache(_CHUNK_CACHE_MAX)
_report_cache = _ReportLRUCache(_REPORT_CACHE_MAX)


# ---------
# Chunking
# ---------
def chunk_text(text: str, max_chars: int = 800, overlap: int = 50) -> list[str]:
    cleaned = clean_text(text).strip()
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

    def generate(self, prompt: str, *, max_output_tokens: int = 500) -> str:
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
# Step 1: per-chunk extraction (not paraphrase summary)
# ----------------------------
def _extract_chunk(llm: LLM, chunk: str) -> str:
    ck = _sha256_utf8(chunk)
    hit = _chunk_extraction_cache.get(ck)
    if hit is not None:
        return hit

    prompt = (
        "Extract legal facts from the text. Output ONLY plain lines (no intro).\n"
        "Rules: max 5 lines; each line max 12 words; fragments/labels OK; "
        "no full sentences; copy terms/numbers when present.\n\n"
        f"{chunk}"
    )
    out = llm.generate(prompt, max_output_tokens=500)
    _chunk_extraction_cache.set(ck, out)
    return out


def _merge_extractions_python(extractions: list[str]) -> str:
    """Dedupe lines; no LLM merge."""
    seen_keys: set[str] = set()
    lines_out: list[str] = []

    for block in extractions:
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            # strip bullet prefixes for dedup stability
            normalized = re.sub(r"^[-*•\d.)]+\s*", "", line).strip().lower()
            if not normalized or normalized in seen_keys:
                continue
            seen_keys.add(normalized)
            lines_out.append(line)

    return "\n".join(lines_out) if lines_out else ""


def _build_compressed_document(llm: LLM, chunks: list[str]) -> str:
    extractions = [_extract_chunk(llm, c) for c in chunks]
    if len(extractions) == 1:
        return extractions[0].strip()
    return _merge_extractions_python(extractions)


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
        out[key] = _coerce_analysis_value(data.get(key))
    return out


def _batched_analysis_prompt(compressed_doc: str) -> str:
    lines = [f'- "{k}": {q}' for k, q in PREDEFINED_QUERIES.items()]
    q_block = "\n".join(lines)
    return (
        "Answer using ONLY <doc>. Extraction style: short labels, numbers, party names; "
        "no paraphrase essays.\n"
        "Per key: max 5 bullet lines; each line max 12 words; fragments OK; "
        "if absent use exactly: Not found in document\n"
        "Return ONLY one JSON object (string values).\n\n"
        f"<doc>\n{compressed_doc}\n</doc>\n\n"
        "Keys:\n"
        f"{q_block}"
    )


def analyze_document(llm: LLM | None = None, compressed_doc: str = "") -> dict[str, str]:
    """
    Runs fixed queries over compressed document (NO RAG) in one model call.
    Never receives the original full document — only merged extractions.
    """
    client = llm or LLM()
    prompt = _batched_analysis_prompt(compressed_doc)
    raw = client.generate(prompt, max_output_tokens=1600)
    parsed = _parse_analysis_json(raw)

    def _all_answered(d: dict[str, str]) -> bool:
        return all((d.get(k) or "").strip() for k in QUERY_KEYS)

    if _all_answered(parsed):
        return {
            k: (parsed.get(k) or "").strip() or "Not found in document"
            for k in QUERY_KEYS
        }

    retry_prompt = (
        _batched_analysis_prompt(compressed_doc)
        + "\n\nJSON only. No fences. No prose outside JSON."
    )
    raw2 = client.generate(retry_prompt, max_output_tokens=1600)
    parsed2 = _parse_analysis_json(raw2)

    merged: dict[str, str] = {}
    for k in QUERY_KEYS:
        v = (parsed.get(k) or parsed2.get(k) or "").strip()
        merged[k] = v if v else "Not found in document"
    return merged


# ----------------------------
# Main pipeline (RAG-free)
# ----------------------------
def run_report_on_text(
    full_text: str,
    llm: LLM | None = None,
    pricing_df: pd.DataFrame | None = None,
) -> dict[str, str]:
    """
    Pipeline: clean → chunk → extract chunks (cached) → Python merge → one JSON analysis.
    Original full text is not sent to the final query — only merged extractions.
    """
    client = llm or LLM()
    cleaned = clean_text(full_text).strip()
    if not cleaned:
        return {k: "No extractable text in the document." for k in QUERY_KEYS}

    doc_key: str | None = None
    if pricing_df is None or pricing_df.empty:
        doc_key = _sha256_utf8(cleaned)
        cached_report = _report_cache.get(doc_key)
        if cached_report is not None:
            return _finalize_report_payload(cached_report)

    chunks = chunk_text(cleaned)
    if not chunks:
        return {k: "No extractable text in the document." for k in QUERY_KEYS}

    compressed_doc = _build_compressed_document(client, chunks)
    results = analyze_document(llm=client, compressed_doc=compressed_doc)

    if pricing_df is not None and not pricing_df.empty:
        # Lazy import: scikit-learn is heavy; document-only analysis should not load it.
        from anomaly_price_detector import analyze_price_history

        try:
            pricing_result = analyze_price_history(pricing_df)
            results["price_anomalies"] = (
                    pricing_result.summary or "No pricing anomaly summary available."
            )
            if pricing_result.warning:
                results["price_anomalies_warning"] = pricing_result.warning
        except Exception as exc:  # noqa: BLE001
            results["price_anomalies"] = (
                f"The analysis failed: {type(exc).__name__}: {exc}"
            )
    finalized = _finalize_report_payload(results)
    if doc_key is not None:
        _report_cache.set(doc_key, dict(finalized))
    return finalized