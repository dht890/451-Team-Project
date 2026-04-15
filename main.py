"""
Minimal FastAPI app: document upload + AI/LLM summary.

Run from this folder, then open the app in the browser (same origin as /analyze):
  uvicorn main:app --reload

Open: http://127.0.0.1:8000/
"""

from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def _is_pdf(upload: UploadFile) -> bool:
    content_type = (upload.content_type or "").lower()
    if content_type in {"application/pdf", "application/x-pdf"}:
        return True
    filename = upload.filename or ""
    return filename.lower().endswith(".pdf")


def _safe_filename(filename: str | None) -> str:
    # Prevent path traversal; keep only the final component.
    return Path(filename or "upload.bin").name

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

    if _is_pdf(file):
        message = f"PDF uploaded: {safe_name} ({size_bytes} bytes)"
    else:
        message = f"Document uploaded: {safe_name} ({size_bytes} bytes)"

    # Server-side verification message (visible in server logs).
    print(message)

    return {
        "message": message,
        "filename": safe_name,
        "bytes": size_bytes,
        "content_type": file.content_type,
        "stored_as": f"/uploads/{safe_name}",
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

