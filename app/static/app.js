"use strict";

const $ = (id) => document.getElementById(id);
const KEY_STORAGE = "visual-search-api-key";

const els = {
  apiKey: $("apiKey"), toggleKey: $("toggleKey"), dropzone: $("dropzone"), fileInput: $("fileInput"),
  preview: $("preview"), dropHint: $("dropHint"), topK: $("topK"), searchBtn: $("searchBtn"),
  error: $("error"), results: $("results"), meta: $("meta"), status: $("status"),
  curlSnippet: $("curlSnippet"), copyCurl: $("copyCurl"),
};

let currentFile = null;
let busy = false;

function storageGet(key) { try { return localStorage.getItem(key) || ""; } catch { return ""; } }
function storageSet(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } }

function el(tag, props = {}, children = []) {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of children) node.append(child);
  return node;
}

function showError(message) {
  els.error.textContent = message;
  els.error.hidden = !message;
}

function updateButton() {
  els.searchBtn.disabled = busy || !currentFile;
  els.searchBtn.textContent = busy ? "Searching…" : "Search";
}

function updateCurl() {
  els.curlSnippet.textContent =
    `curl -X POST "${location.origin}/api/v1/search?top_k=${els.topK.value}" \\\n` +
    `  -H "X-API-Key: YOUR_API_KEY" \\\n` +
    `  -F "file=@photo.jpg"`;
}

async function loadHealth() {
  try {
    const res = await fetch("/api/v1/health");
    const body = await res.json();
    els.status.textContent = `${body.indexed.toLocaleString()} images · ${body.index_type}`;
    els.status.className = "pill ok";
  } catch {
    els.status.textContent = "API unreachable";
    els.status.className = "pill bad";
  }
}

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    showError("That file is not an image.");
    return;
  }
  currentFile = file;
  showError("");
  els.preview.src = URL.createObjectURL(file);
  els.preview.hidden = false;
  els.dropHint.hidden = true;
  updateButton();
  search();
}

function renderSkeleton(count) {
  els.results.replaceChildren(...Array.from({ length: count }, () => el("div", { className: "skeleton" })));
}

function renderResults(body) {
  els.meta.textContent = `${body.results.length} of ${body.total_indexed.toLocaleString()} · ${Math.round(body.took_ms)} ms`;
  if (!body.results.length) {
    els.results.replaceChildren(el("p", { className: "empty", textContent: "The collection is empty." }));
    return;
  }
  const cards = body.results.map((r) => {
    const pct = Math.max(0, Math.min(100, r.score * 100));
    const img = el("img", { src: r.thumbnail_url, alt: r.filename, loading: "lazy" });
    // The free tunnel occasionally drops a request; retry each thumbnail once.
    img.addEventListener("error", () => setTimeout(() => { img.src = `${r.thumbnail_url}&retry=1`; }, 400), { once: true });
    const link = el("a", { href: r.image_url, target: "_blank", rel: "noopener", title: "Open full image" }, [
      img, el("span", { className: "rank", textContent: `#${r.rank}` }),
    ]);
    const bar = el("div", { className: "bar" }, [el("span", { style: `width:${pct.toFixed(1)}%` })]);
    return el("article", { className: "card" }, [
      link,
      el("div", { className: "card-body" }, [
        el("div", { className: "name", textContent: r.name || r.filename, title: r.name || r.filename }),
        el("div", { className: "sub", textContent: [r.brand, r.price_usd != null ? `$${r.price_usd.toFixed(2)}` : null].filter(Boolean).join(" · ") }),
        el("div", { className: "score-row" }, [bar, el("span", { textContent: `${pct.toFixed(1)}%` })]),
      ]),
    ]);
  });
  els.results.replaceChildren(...cards);
}

async function search() {
  if (!currentFile || busy) return;
  const key = els.apiKey.value.trim();
  if (!key) {
    showError("Enter your API key first.");
    els.apiKey.focus();
    return;
  }
  busy = true;
  updateButton();
  showError("");
  renderSkeleton(Number(els.topK.value));
  els.meta.textContent = "";

  const form = new FormData();
  form.append("file", currentFile);
  try {
    const res = await fetch(`/api/v1/search?top_k=${els.topK.value}`, {
      method: "POST",
      headers: { "X-API-Key": key },
      body: form,
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail || res.statusText);
      showError(`${res.status}: ${detail}`);
      els.results.replaceChildren(el("p", { className: "empty", textContent: "No results." }));
      return;
    }
    renderResults(body);
  } catch (err) {
    showError(`Network error: ${err.message}`);
    els.results.replaceChildren();
  } finally {
    busy = false;
    updateButton();
  }
}

// --- wiring ---------------------------------------------------------------

els.apiKey.value = storageGet(KEY_STORAGE);
els.apiKey.addEventListener("input", () => storageSet(KEY_STORAGE, els.apiKey.value.trim()));
els.apiKey.addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
els.toggleKey.addEventListener("click", () => {
  const hidden = els.apiKey.type === "password";
  els.apiKey.type = hidden ? "text" : "password";
  els.toggleKey.textContent = hidden ? "Hide" : "Show";
});

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
});
els.fileInput.addEventListener("change", () => { setFile(els.fileInput.files[0]); els.fileInput.value = ""; });
["dragenter", "dragover"].forEach((t) => els.dropzone.addEventListener(t, (e) => {
  e.preventDefault(); els.dropzone.classList.add("over");
}));
["dragleave", "drop"].forEach((t) => els.dropzone.addEventListener(t, (e) => {
  e.preventDefault(); els.dropzone.classList.remove("over");
}));
els.dropzone.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
document.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) setFile(item.getAsFile());
});

els.searchBtn.addEventListener("click", search);
els.topK.addEventListener("change", () => { updateCurl(); search(); });
els.copyCurl.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(els.curlSnippet.textContent);
    els.copyCurl.textContent = "Copied";
    setTimeout(() => { els.copyCurl.textContent = "Copy"; }, 1500);
  } catch { /* clipboard blocked */ }
});

updateCurl();
updateButton();
loadHealth();
