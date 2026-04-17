#main.py
"""
Minimal FastAPI app: document upload + AI/LLM summary.

Run from this folder, then open the app in the browser (same origin as /analyze):
  uvicorn main:app --reload

Open: http://127.0.0.1:8000/
"""
 
from pathlib import Path

from docx import Document as DocxDocument
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader

from report import LLM, run_report_on_text

def _safe_filename(filename: str | None) -> str:
    # Prevent path traversal; keep only the final component.
    return Path(filename or "upload.bin").name


def _extract_text_from_path(path: Path) -> tuple[str, str | None]:
    """
    Return (text, warning). Warning is set when text is empty or extraction is partial.
    """
    suffix = path.suffix.lower()
    if suffix == ".txt":
        raw = path.read_text(encoding="utf-8", errors="replace")
        return raw, None if raw.strip() else "File appears empty."

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        parts: list[str] = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                parts.append(t)
        text = "\n\n".join(parts).strip()
        if not text:
            return "", "Could not extract text from PDF (may be scanned or encrypted)."
        return text, None

    if suffix == ".docx":
        doc = DocxDocument(str(path))
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n\n".join(paras).strip()
        if not text:
            return "", "No text found in DOCX."
        return text, None

    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return "", "Image uploads are not OCR-processed; use PDF or text for analysis."

    return "", f"Unsupported type for text extraction: {suffix or '(no extension)'}"


STATIC_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = STATIC_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Document analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    safe_name = _safe_filename(file.filename)
    dest_path = UPLOAD_DIR / safe_name

    # Save upload to disk (streaming in chunks).
    size_bytes = 0
    with dest_path.open("wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size_bytes += len(chunk)
            f.write(chunk)

    kind = "PDF" if dest_path.suffix.lower() == ".pdf" else "Document"
    message = f"{kind} uploaded: {safe_name} ({size_bytes} bytes)"

    # Server-side verification message (visible in server logs).
    print(message)

    extracted, extraction_warning = _extract_text_from_path(dest_path)
    report: dict[str, str] | None = None
    summary_parts: list[str] = []
    llm_model: str | None = None

    if extracted.strip():
        try:
            llm = LLM()
            llm_model = llm.model
            report = run_report_on_text(extracted, llm=llm)
            for section_key, answer in report.items():
                title = section_key.replace("_", " ").title()
                summary_parts.append(f"## {title}\n{answer}")
        except Exception as exc:  # noqa: BLE001 — surface LLM/config errors to client
            extraction_warning = (
                f"Report generation failed ({type(exc).__name__}). "
            )
            print(extraction_warning, exc)
    else:
        extraction_warning = extraction_warning or "No text extracted; report skipped."

    summary = "\n\n".join(summary_parts) if summary_parts else ""

    return {
        "message": message,
        "filename": safe_name,
        "bytes": size_bytes,
        "content_type": file.content_type,
        "stored_as": f"/uploads/{safe_name}",
        "extracted_text_chars": len(extracted),
        "estimated_tokens": len(extracted) // 4,  # Rough estimate
        "summary": summary,
        "report": report,
        "warning": extraction_warning,
        "model": llm_model,
    }


# Serve the UI entrypoint explicitly. This avoids surprises if the server is
# started from a different working directory or if StaticFiles resolution changes.
@app.get("/", include_in_schema=False)
def ui_index():
    return FileResponse(STATIC_DIR / "index.html")


# Serve top-level assets referenced by index.html.
@app.get("/script.js", include_in_schema=False)
def ui_script():
    return FileResponse(STATIC_DIR / "script.js")


@app.get("/styles.css", include_in_schema=False)
def ui_styles():
    return FileResponse(STATIC_DIR / "styles.css")


# Serve the UI + uploaded files.
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

