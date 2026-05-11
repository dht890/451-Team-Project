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
const pricingInput = document.getElementById("pricingInput");
const pricingFileStatusEl = document.getElementById("pricingFileStatus");

const analysisBadgeEl = document.getElementById("analysisBadge");
const analysisMetaEl = document.getElementById("analysisMeta");
const analysisWarningEl = document.getElementById("analysisWarning");
const analysisReportEl = document.getElementById("analysisReport");

function ensurePort8000Banner() {
  if (window.location.protocol === "file:") return;
  if (window.location.port === "8000") return;

  const targetUrl = `${window.location.protocol}//${window.location.hostname}:8000`;

  const banner = document.createElement("div");
  banner.className = "dev-banner";

  const text = document.createElement("div");
  text.textContent = `This page is running on port ${window.location.port || "(default)"}; the backend expects port 8000.`;
  text.style.marginBottom = "10px";
  banner.append(text);

  const p = document.createElement("p");
  p.textContent = "Run: uvicorn main:app --reload";
  banner.append(p);

  const p2 = document.createElement("p");
  p2.textContent = "Then open port 8000 so upload can reach /analyze.";
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

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function setBadge(mode, label) {
  if (!analysisBadgeEl) return;
  analysisBadgeEl.textContent = label;
  analysisBadgeEl.classList.remove("badge--muted", "badge--active", "badge--error");
  if (mode === "active") analysisBadgeEl.classList.add("badge--active");
  else if (mode === "error") analysisBadgeEl.classList.add("badge--error");
  else analysisBadgeEl.classList.add("badge--muted");
}

function clearMetaWarning() {
  if (analysisMetaEl) {
    analysisMetaEl.hidden = true;
    analysisMetaEl.innerHTML = "";
  }
  if (analysisWarningEl) {
    analysisWarningEl.hidden = true;
    analysisWarningEl.textContent = "";
    analysisWarningEl.classList.remove("analysis-alert--error");
  }
}

function updatePricingStatus() {
  if (!pricingFileStatusEl) return;
  const pricingFile = pricingInput?.files?.[0];
  if (pricingFile) {
    pricingFileStatusEl.hidden = false;
    pricingFileStatusEl.textContent = `Selected CSV: ${pricingFile.name}`;
  } else {
    pricingFileStatusEl.hidden = false;
    pricingFileStatusEl.textContent = "No CSV selected.";
  }
}

/** Turn plain multi-line answers into safe HTML paragraphs or a bullet list. */
function formatAnswerHtml(raw) {
  const text = String(raw ?? "").trim();
  if (!text) {
    return `<p>${escapeHtml("Not found in document.")}</p>`;
  }

  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  const bulletLike = /^[\-\*•]\s+|^\d+[.)]\s+/;

  const allBullets = lines.length > 0 && lines.every((l) => bulletLike.test(l));
  if (allBullets) {
    const items = lines.map((line) =>
      escapeHtml(line.replace(bulletLike, "").trim()),
    );
    return `<ul>${items.map((t) => `<li>${t}</li>`).join("")}</ul>`;
  }

  return lines.map((line) => `<p>${escapeHtml(line)}</p>`).join("");
}

/** Stable section order matching backend PREDEFINED_QUERIES keys. */
function renderReportCards(report) {
  if (!analysisReportEl) return;

  const order = [
    "termination",
    "liability",
    "payment",
    "confidentiality",
    "risks",
    "pricing_risk",
    "price_anomalies",
    "price_anomalies_warning",
  ];

  const frag = document.createDocumentFragment();

  let any = false;
  for (const key of order) {
    if (!Object.prototype.hasOwnProperty.call(report, key)) continue;
    any = true;
    const label = key
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());

    const card = document.createElement("article");
    card.className = "report-card";

    const h3 = document.createElement("h3");
    h3.className = "report-card__title";
    h3.textContent = label;

    const body = document.createElement("div");
    body.className = "report-card__body";
    body.innerHTML = formatAnswerHtml(report[key]);

    card.append(h3, body);
    frag.append(card);
  }

  analysisReportEl.innerHTML = "";
  if (!any) {
    const fallback = document.createElement("div");
    fallback.className = "report-empty";
    fallback.textContent = "No structured report returned.";
    analysisReportEl.append(fallback);
  } else {
    analysisReportEl.append(frag);
  }
}

function renderMarkdownishSummary(summary) {
  if (!analysisReportEl) return;
  const text = String(summary ?? "").trim();
  if (!text) {
    const empty = document.createElement("div");
    empty.className = "report-empty";
    empty.textContent = "No report returned.";
    analysisReportEl.innerHTML = "";
    analysisReportEl.append(empty);
    return;
  }

  /** Split summary on ## headings from server markdown. */
  const sections = [];
  const lines = text.split("\n");
  let currentTitle = null;
  let buf = [];

  function flush() {
    if (!currentTitle && buf.length === 0) return;
    sections.push({
      title: currentTitle || "Summary",
      body: buf.join("\n").trim(),
    });
    currentTitle = null;
    buf = [];
  }

  for (const line of lines) {
    const m = /^##\s+(.+)$/.exec(line.trim());
    if (m) {
      flush();
      currentTitle = m[1].trim();
    } else {
      buf.push(line);
    }
  }
  flush();

  analysisReportEl.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const { title, body } of sections) {
    const card = document.createElement("article");
    card.className = "report-card";
    const h3 = document.createElement("h3");
    h3.className = "report-card__title";
    h3.textContent = title;
    const div = document.createElement("div");
    div.className = "report-card__body";
    div.innerHTML = formatAnswerHtml(body);
    card.append(h3, div);
    frag.append(card);
  }
  analysisReportEl.append(frag);
}

function setLoadingState() {
  setBadge("active", "Working");
  clearMetaWarning();

  if (analysisReportEl) {
    analysisReportEl.innerHTML = "";
    const wrap = document.createElement("div");
    wrap.className = "report-loading";
    wrap.innerHTML =
      '<span class="report-loading__spin" aria-hidden="true"></span><span>Analyzing document…</span>';
    wrap.setAttribute("role", "status");
    analysisReportEl.append(wrap);
  }
}

function setIdleState() {
  setBadge("muted", "Ready");
  clearMetaWarning();
  if (analysisReportEl) {
    analysisReportEl.innerHTML = "";
    const p = document.createElement("p");
    p.className = "report-empty";
    p.id = "analysisEmpty";
    p.textContent =
      "Upload a document(s) to see termination, liability, payment, confidentiality, risk notes, and optionally pricing anomaly findings.";
    analysisReportEl.append(p);
  }
}

function setErrorState(message) {
  setBadge("error", "Issue");
  clearMetaWarning();

  if (analysisWarningEl) {
    analysisWarningEl.hidden = false;
    analysisWarningEl.classList.add("analysis-alert--error");
    analysisWarningEl.textContent = message;
  }

  if (analysisReportEl) {
    analysisReportEl.innerHTML = "";
    const hint = document.createElement("div");
    hint.className = "report-empty";
    hint.textContent = "Fix the warning above or check that the API is running.";
    analysisReportEl.append(hint);
  }
}

function renderAnalysisSuccess(data) {
  const message = String(data.message ?? "").trim();
  const summary = String(data.summary ?? "").trim();
  const warning = String(data.warning ?? "").trim();
  const model = String(data.model ?? "").trim();
  const pricingFile = pricingInput?.files?.[0];
  const extractedChars = data.extracted_text_chars ?? data.extractedTextChars;

  clearMetaWarning();
  setBadge("active", "Done");

  if (analysisMetaEl) {
    const chips = [];

    chips.push(document.createElement("span"));
    chips[chips.length - 1].className = "chip";
    chips[chips.length - 1].innerHTML = `<strong>File</strong> ${escapeHtml(message || "(upload)")}`;

    if (model) {
      const m = document.createElement("span");
      m.className = "chip";
      m.innerHTML = `<strong>Model</strong> ${escapeHtml(model)}`;
      chips.push(m);
    }

    if (pricingFile) {
      const p = document.createElement("span");
      p.className = "chip";
      p.innerHTML = `<strong>Pricing CSV</strong> ${escapeHtml(pricingFile.name)}`;
      chips.push(p);
    }

    if (typeof extractedChars === "number") {
      const c = document.createElement("span");
      c.className = "chip";
      c.innerHTML = `<strong>Extracted</strong> ${extractedChars.toLocaleString()} chars`;
      chips.push(c);
    }

    analysisMetaEl.innerHTML = "";
    for (const el of chips) analysisMetaEl.append(el);
    analysisMetaEl.hidden = chips.length === 0;
  }

  if (analysisWarningEl && warning) {
    analysisWarningEl.hidden = false;
    analysisWarningEl.classList.remove("analysis-alert--error");
    analysisWarningEl.textContent = warning;
  }

  const hasReport =
    data.report &&
    typeof data.report === "object" &&
    Object.keys(data.report).length > 0;

  if (hasReport) {
    renderReportCards(data.report);
  } else {
    renderMarkdownishSummary(summary || "No report returned.");
  }
}

function isPdfFile(file) {
  const mime = (file.type || "").toLowerCase();
  if (mime === "application/pdf" || mime === "application/x-pdf") return true;
  return file.name?.toLowerCase().endsWith(".pdf") ?? false;
}

function setPreview(type, data) {
  revokeUrl();

  preview.removeAttribute("src");
  preview.alt = "";
  preview.hidden = true;
  preview.style.display = "none";

  pdfPreview.removeAttribute("src");
  pdfPreview.hidden = true;
  pdfPreview.style.display = "none";

  previewPlaceholder.hidden = true;

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
    previewPlaceholder.hidden = false;
  }
}

function clearPreview() {
  setPreview("none");
}

async function analyzeWithBackend(file, pricingFile = null) {
  const formData = new FormData();
  formData.append("file", file);
  if (pricingFile) {
    formData.append("pricing_file", pricingFile);
  }

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

let currentFileToken = 0;

function runCurrentAnalysis() {
  const file = input?.files?.[0];
  const pricingFile = pricingInput?.files?.[0] || null;
  const token = ++currentFileToken;

  clearPreview();

  if (!file) {
    setIdleState();
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

  setLoadingState();

  analyzeWithBackend(file, pricingFile)
    .then((data) => {
      if (token !== currentFileToken) return;

      if (!data || typeof data !== "object") {
        renderAnalysisSuccess({
          message: "",
          summary: String(data),
          warning: "",
          model: "",
          report: null,
        });
        return;
      }

      renderAnalysisSuccess(data);
    })
    .catch(() => {
      if (token !== currentFileToken) return;
      setErrorState(
        "Could not reach the analyzer. Start the API with uvicorn main:app --reload and open this page from port 8000.",
      );
    });
}

input?.addEventListener("change", () => {
  runCurrentAnalysis();
});

pricingInput?.addEventListener("change", () => {
  updatePricingStatus();

  if (input?.files?.[0]) {
    runCurrentAnalysis();
  }
});

updatePricingStatus();