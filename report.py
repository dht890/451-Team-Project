#report.py
"""
Legal document report: single LLM call over cleaned full text → JSON.
No chunking, no merge step. Report LRU avoids repeat calls for identical text.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import OrderedDict

import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

PREDEFINED_QUERIES = {
    "termination": "What are the termination conditions or exit clauses?",
    "liability": "What liability limitations or risk allocations exist?",
    "payment": "What are the payment terms, fees, and billing structure?",
    "confidentiality": "What confidentiality or NDA clauses are included?",
    "risks": "What unusual, risky, or one-sided clauses exist?",
}

QUERY_KEYS = tuple(PREDEFINED_QUERIES.keys())

_REPORT_CACHE_MAX = 64
# Bump when pipeline/prompt changes so stale LRU entries are not reused.
_REPORT_CACHE_SALT = "v4-json-schema-preprocess"

# Gemini structured output (exact keys, valid JSON).
def _report_response_schema() -> dict[str, object]:
    return {
        "type": "OBJECT",
        "properties": {k: {"type": "STRING"} for k in QUERY_KEYS},
        "required": list(QUERY_KEYS),
    }


import re

def clean_text(text: str) -> str:
    """
    Preserve structure (newlines) while still cleaning noise.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove excessive spaces but KEEP newlines
    text = re.sub(r"[ \t]+", " ", text)

    # optional: normalize multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


LEGAL_KEYWORDS = [
    "termination",
    "terminate",
    "cancellation",
    "end of",
    "expiration",
    "liability",
    "liable",
    "indemnity",
    "indemnify",
    "damages",
    "payment",
    "fee",
    "billing",
    "charges",
    "compensation",
    "confidential",
    "non-disclosure",
    "nda",
    "privacy",
    "arbitration",
    "dispute",
    "governing law",
    "breach",
    "violation",
    "rights",
    "obligations",
    "agreement",
    "terms",
    "services",
]


def _looks_legal(line: str) -> bool:
    l = line.lower()
    return any(k in l for k in LEGAL_KEYWORDS)


def preprocess_document(text: str, context_window: int = 2) -> str:
    """
    Deterministic pre-filter: keep sentence-like units with legal signals plus
    ±context_window neighbors. Targets large token reduction before the LLM.
    """
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    parts = re.split(r"(?<=[\.\;\:])\s+", text)
    if not parts or (len(parts) == 1 and not parts[0].strip()):
        return text.strip()

    kept: set[int] = set()
    for i, p in enumerate(parts):
        if _looks_legal(p):
            lo = max(0, i - context_window)
            hi = min(len(parts), i + context_window + 1)
            for j in range(lo, hi):
                kept.add(j)

    if not kept:
        return text.strip()

    filtered = [parts[i] for i in sorted(kept)]
    return " ".join(filtered)


def _sha256_utf8(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


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


_report_cache = _ReportLRUCache(_REPORT_CACHE_MAX)


# ------------
# LLM wrapper
# ------------
class LLM:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash")
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str, *, max_output_tokens: int = 300) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "temperature": 0.1,
                "top_p": 0.9,
                "max_output_tokens": max_output_tokens,
            },
        )
        return (response.text or "").strip()

    def generate_report_payload(
    self, prompt: str, *, max_output_tokens: int
    ) -> tuple[object | None, str]:
        """
        Returns:
            (parsed_json_if_available, raw_text)
        Also prints token usage when available.
        """

        base_cfg: dict[str, object] = {
            "temperature": 0.1,
            "top_p": 0.9,
            "max_output_tokens": max_output_tokens,
        }

        # ---- Try structured JSON mode first ----
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    **base_cfg,
                    "response_mime_type": "application/json",
                    "response_schema": _report_response_schema(),
                },
            )
        except Exception:
            # fallback to plain mode if schema fails
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=base_cfg,
            )

        # ---- Extract outputs ----
        parsed = getattr(response, "parsed", None)
        raw = (response.text or "").strip()

        # ---- TOKEN USAGE LOGGING ----
        usage = getattr(response, "usage_metadata", None)
        if usage:
            print("\n===== TOKEN USAGE =====")
            print(f"prompt_tokens: {getattr(usage, 'prompt_token_count', None)}")
            print(f"output_tokens: {getattr(usage, 'candidates_token_count', None)}")
            print(f"total_tokens: {getattr(usage, 'total_token_count', None)}")
            print("=======================\n")
        else:
            print("\n[Token usage not available for this response]\n")

        return parsed, raw


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


def _coerce_to_answer_str(val: object) -> str:
    """Model may return strings or lists of bullets; normalize to one string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (int, float, bool)):
        return str(val).strip()
    if isinstance(val, list):
        lines = [_coerce_to_answer_str(x) for x in val]
        return "\n".join(line for line in lines if line)
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return str(val).strip()


def _report_from_model_dict(data: dict[object, object]) -> dict[str, str]:
    """Map model JSON to QUERY_KEYS; tolerate different key casing."""
    keymap: dict[str, object] = {}
    for k, v in data.items():
        if isinstance(k, str):
            keymap[k.strip().lower()] = v
    return {qk: _coerce_to_answer_str(keymap.get(qk.lower())) for qk in QUERY_KEYS}


def _parse_analysis_json(raw: str) -> dict[str, str]:
    """Parse JSON from model text; tolerate fences and leading prose."""
    cleaned = _strip_code_fence(raw.strip())
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

    if not isinstance(data, dict):
        return {k: "" for k in QUERY_KEYS}
    return _report_from_model_dict(data)


def _single_pass_prompt(doc: str) -> str:
    keys_example = ", ".join(f'"{k}": "…"' for k in QUERY_KEYS)

    return f"""
Read <doc> and extract legal clauses into structured JSON.

Definitions:
- termination: termination, cancellation, expiration, ending agreement
- liability: liability, damages, indemnity, limitation of liability
- payment: fees, billing, charges, compensation
- confidentiality: confidentiality, NDA, non-disclosure, privacy
- risks: penalties, unilateral rights, unusual obligations

Rules:
- Prefer copying exact phrases from the document
- Use short bullet-style lines (newline separated)
- Max ~3 bullets per field
- Include partial matches if relevant
- Only use "Not found in document" if absolutely nothing exists

Return EXACT JSON format:
{{{keys_example}}}

<doc>
{doc}
</doc>
"""


def _normalize_report(parsed: dict[str, str]) -> dict[str, str]:
    return {k: (parsed.get(k) or "").strip() or "Not found in document" for k in QUERY_KEYS}


def extract_report_single_pass(llm: LLM, doc: str) -> dict[str, str]:
    """One LLM call: full document → structured report."""
    prompt = _single_pass_prompt(doc)
    parsed_obj, raw = llm.generate_report_payload(prompt, max_output_tokens=3000)

    print("\n===== RAW MODEL OUTPUT =====")
    print(raw)
    print("===== END RAW OUTPUT =====\n")
    
    if isinstance(parsed_obj, dict) and parsed_obj:
        fields = _report_from_model_dict(parsed_obj)
    else:
        fields = _parse_analysis_json(raw)
    return _normalize_report(fields)


def analyze_document(llm: LLM | None = None, compressed_doc: str = "") -> dict[str, str]:
    """
    Back-compat alias: `compressed_doc` is treated as the full document text to analyze.
    Single LLM call, no chunking.
    """
    client = llm or LLM()
    return extract_report_single_pass(client, compressed_doc)


# ----------------------------
# Main pipeline
# ----------------------------
def run_report_on_text(full_text: str, llm: LLM | None = None) -> dict[str, str]:
    """
    clean → cache check → one LLM call (full doc) → JSON report.
    """
    client = llm or LLM()
    cleaned = clean_text(full_text)
    cleaned = preprocess_document(cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return {k: "No extractable text in the document." for k in PREDEFINED_QUERIES}

    doc_key = _sha256_utf8(_REPORT_CACHE_SALT + "\n" + cleaned)
    cached_report = _report_cache.get(doc_key)
    if cached_report is not None:
        return cached_report

    report = extract_report_single_pass(client, cleaned)
    _report_cache.set(doc_key, report)
    return report
