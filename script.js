//script.js
/** Same origin when the page is served by FastAPI; fallback for file:// opens. */
const API_BASE =
  window.location.protocol === "file:"
    ? "http://127.0.0.1:8000"
    : "";

const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const pdfPreview = document.getElementById("pdfPreview");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const analysisEl = document.getElementById("analysis");
const analysisOutputEl = document.getElementById("analysisOutput");

function ensurePort8000Banner() {
  // If served from a dev server on the wrong port, guide users to 8000.
  // (For file:// we already hardcode API_BASE to 8000.)
  if (window.location.protocol === "file:") return;
  if (window.location.port === "8000") return;

  const targetUrl = `${window.location.protocol}//${window.location.hostname}:8000`;

  const banner = document.createElement("div");
  banner.style.margin = "12px 0";
  banner.style.padding = "12px 14px";
  banner.style.border = "1px solid rgba(255, 170, 0, 0.45)";
  banner.style.borderRadius = "10px";
  banner.style.background = "rgba(255, 170, 0, 0.08)";

  const text = document.createElement("div");
  text.textContent = `This page is running on port ${window.location.port || "(default)"}; the backend expects port 8000.`;
  text.style.marginBottom = "10px";
  banner.append(text);

  const p = document.createElement("p");
  p.textContent = "Run the command: uvicorn main:app --reload";
  banner.append(p);

  const p2 = document.createElement("p");
  p2.textContent = "Then click the button below to go to the correct port.";
  banner.append(p2);

  const btn = document.createElement("button");
  btn.type = "button";
  btn.textContent = "Go to port 8000";
  btn.addEventListener("click", () => {
    window.location.assign(targetUrl);
  });
  banner.append(btn);

  const app = document.querySelector("main.app");
  const header = document.querySelector(".app__header");
  if (app && header && header.parentElement === app) {
    header.insertAdjacentElement("afterend", banner);
  } else {
    document.body.insertAdjacentElement("afterbegin", banner);
  }
}

ensurePort8000Banner();

/** Revoked when the file changes so blob: URLs don't leak. */
let previewObjectUrl = null;

function revokeUrl() {
  if (previewObjectUrl) {
    URL.revokeObjectURL(previewObjectUrl);
    previewObjectUrl = null;
  }
}

function isPdfFile(file) {
  const mime = (file.type || "").toLowerCase();
  if (mime === "application/pdf" || mime === "application/x-pdf") return true;
  return file.name?.toLowerCase().endsWith(".pdf") ?? false;
}

function setOutput(text) {
  if (analysisOutputEl) {
    analysisOutputEl.textContent = text;
  } else {
    analysisEl.textContent = text;
  }
}

/** Stable section order matching backend PREDEFINED_QUERIES keys. */
function formatReportFromJson(report) {
  if (!report || typeof report !== "object") return "";
  const order = [
    "termination",
    "liability",
    "payment",
    "confidentiality",
    "risks",
  ];
  const parts = [];
  for (const key of order) {
    if (!Object.prototype.hasOwnProperty.call(report, key)) continue;
    const label = key
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
    parts.push(`## ${label}\n${String(report[key] ?? "").trim()}`);
  }
  return parts.join("\n\n").trim();
}

function setPreview(type, data) {
  // Always start clean
  revokeUrl();

  // Reset everything
  preview.removeAttribute("src");
  preview.alt = "";
  preview.hidden = true;
  preview.style.display = "none";

  pdfPreview.removeAttribute("src");
  pdfPreview.hidden = true;
  pdfPreview.style.display = "none";

  previewPlaceholder.hidden = true;

  // Render based on type
  if (type === "image") {
    preview.src = data.url;
    preview.alt = data.fileName
      ? `Preview: ${data.fileName}`
      : "Selected image preview";
    preview.hidden = false;
    preview.style.display = "block";
  } else if (type === "pdf") {
    previewObjectUrl = URL.createObjectURL(data.file);
    pdfPreview.src = previewObjectUrl;
    pdfPreview.hidden = false;
    pdfPreview.style.display = "block";
  } else {
    // fallback (no preview)
    previewPlaceholder.hidden = false;
  }
}

function clearPreview() {
  setPreview("none");
}
/**
 * Optional: POST the file to the FastAPI backend when it is running.
 * Safe to fail silently when the server is off (local file preview still works).
 */
async function analyzeWithBackend(file) {
  const formData = new FormData();
  formData.append("file", file);

  const analyzeUrl = API_BASE ? `${API_BASE}/analyze` : "/analyze";
  const response = await fetch(analyzeUrl, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Analyze failed: ${response.status}`);
  }

  return response.json();
}

/** Incremented on each selection to ignore stale async callbacks. */
let currentFileToken = 0;
input.addEventListener("change", () => {
  const file = input.files?.[0];
  const token = ++currentFileToken;

  clearPreview();

  if (!file) {
    setOutput("—");
    return;
  }

  if (isPdfFile(file)) {
    setPreview("pdf", { file });
  } else if ((file.type || "").toLowerCase().startsWith("image/")) {
    const reader = new FileReader();

    reader.addEventListener("load", () => {
      if (token !== currentFileToken) return;

      if (typeof reader.result === "string") {
        setPreview("image", {
          url: reader.result,
          fileName: file.name,
        });
      }
    });

    reader.readAsDataURL(file);
  } else {
    setPreview("none");
  }

  setOutput("Analyzing…");

  analyzeWithBackend(file)
    .then((data) => {
      if (token !== currentFileToken) return;

      if (!data || typeof data !== "object") {
        setOutput(String(data));
        return;
      }

      const message = String(data.message ?? "").trim();
      const summary = String(data.summary ?? "").trim();
      const warning = String(data.warning ?? "").trim();
      const model = String(data.model ?? "").trim();
      const extractedChars = data.extracted_text_chars ?? data.extractedTextChars;
      const reportBody =
        formatReportFromJson(data.report) || summary || "No report returned.";

      const headerLines = [
        message ? message : null,
        model ? `Model: ${model}` : null,
        typeof extractedChars === "number"
          ? `Extracted text: ${extractedChars.toLocaleString()} chars`
          : null,
        warning ? `Warning: ${warning}` : null,
      ].filter(Boolean);

      setOutput(
        (headerLines.length ? headerLines.join("\n") + "\n\n" : "") + reportBody
      );
    })
    .catch(() => {
      if (token !== currentFileToken) return;
      setOutput("Failed to analyze the file. Is the backend running?");
    });
});