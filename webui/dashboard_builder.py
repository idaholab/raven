"""
Generate a compact "output dashboard" HTML page for a RAVEN run directory.

This is intentionally generic: it scans a run output folder for generated
outputs and renders a single page that embeds them via the webui file
endpoint (so relative paths do not need to work).
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class DashboardItem:
  path: str
  label: str
  kind: str
  size: int


def discover_outputs(run_dir: Path) -> List[DashboardItem]:
  base = run_dir.resolve()
  items: List[DashboardItem] = []
  if not base.exists():
    return []
  for path in sorted(base.rglob("*")):
    if path.is_dir() or path.name.startswith("."):
      continue
    try:
      rel = str(path.relative_to(base))
    except Exception:
      continue
    name = rel.lower()
    if name.endswith("dashboard.html") or "dashboard_compact" in name:
      continue
    items.append(
      DashboardItem(
        path=rel,
        label=Path(rel).name,
        kind=_classify_kind(path),
        size=_safe_size(path),
      )
    )
  return items


def _classify_kind(path: Path) -> str:
  ext = path.suffix.lower()
  if ext in {".html", ".htm"}:
    return "html"
  if ext in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
    return "image"
  if ext in {".pdf"}:
    return "pdf"
  if ext in {".csv", ".txt", ".xml", ".json", ".log", ".yml", ".yaml"}:
    return "text"
  return "file"


def _safe_size(path: Path) -> int:
  try:
    return path.stat().st_size
  except OSError:
    return 0


def build_compact_dashboard(
  job_id: str,
  run_dir: Path,
  title: str = "Output Dashboard",
  items: Optional[Iterable[DashboardItem]] = None,
) -> str:
  resolved = list(items) if items is not None else discover_outputs(run_dir)
  safe_title = escape(title)
  encoded_items = ",\n".join(
    (
      f'{{path:"{escape(item.path)}",label:"{escape(item.label)}",'
      f'kind:"{escape(item.kind)}",size:{item.size}}}'
    )
    for item in resolved
  )
  # Use the file endpoint for every embedded plot so the dashboard does not rely on
  # relative URL resolution within query-string URLs.
  return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{safe_title}</title>
    <style>
      :root {{
        --bg: #0b1220;
        --panel: #121a2b;
        --panel2: #0f1726;
        --text: #e7edf7;
        --muted: #a6b2c7;
        --accent: #7dd3fc;
        --border: rgba(231, 237, 247, 0.12);
      }}
      * {{ box-sizing: border-box; }}
      html, body {{ height: 100%; }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
        background: radial-gradient(circle at 10% 10%, #1b325a, var(--bg));
      }}
      header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding: 14px 18px;
        border-bottom: 1px solid var(--border);
        background: rgba(11, 18, 32, 0.65);
        backdrop-filter: blur(12px);
        position: sticky;
        top: 0;
        z-index: 20;
      }}
      header h1 {{
        margin: 0;
        font-size: 16px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: var(--muted);
      }}
      header .meta {{
        font-size: 12px;
        color: var(--muted);
      }}
      .layout {{
        display: grid;
        grid-template-columns: 320px minmax(0, 1fr);
        height: calc(100% - 54px);
      }}
      aside {{
        border-right: 1px solid var(--border);
        background: rgba(18, 26, 43, 0.72);
        padding: 12px;
        overflow: auto;
      }}
      main {{
        padding: 14px;
        overflow: auto;
      }}
      .search {{
        display: flex;
        gap: 10px;
        align-items: center;
        margin-bottom: 10px;
      }}
      input[type="search"] {{
        width: 100%;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: rgba(15, 23, 38, 0.8);
        color: var(--text);
        padding: 10px 12px;
      }}
      .list {{
        display: flex;
        flex-direction: column;
        gap: 8px;
      }}
      .list button {{
        text-align: left;
        border: 1px solid rgba(231, 237, 247, 0.1);
        border-radius: 12px;
        background: rgba(15, 23, 38, 0.55);
        color: var(--text);
        padding: 10px 10px;
        cursor: pointer;
      }}
      .list button:hover {{
        border-color: rgba(125, 211, 252, 0.5);
      }}
      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
        gap: 14px;
      }}
      .card {{
        border: 1px solid rgba(231, 237, 247, 0.12);
        border-radius: 16px;
        background: rgba(18, 26, 43, 0.65);
        overflow: hidden;
        min-height: 340px;
        display: flex;
        flex-direction: column;
      }}
      .card__header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 10px 12px;
        border-bottom: 1px solid rgba(231, 237, 247, 0.10);
        background: rgba(11, 18, 32, 0.35);
      }}
      .card__title {{
        font-size: 12px;
        color: var(--text);
        overflow-wrap: anywhere;
      }}
      .card__actions a {{
        font-size: 12px;
        color: var(--accent);
        text-decoration: none;
      }}
      .card__actions a:hover {{
        text-decoration: underline;
      }}
      iframe {{
        border: 0;
        width: 100%;
        flex: 1 1 auto;
        min-height: 320px;
        background: rgba(11, 18, 32, 0.35);
      }}
      img.asset {{
        width: 100%;
        height: 100%;
        object-fit: contain;
        background: rgba(11, 18, 32, 0.35);
      }}
      .file-preview {{
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding: 12px;
        color: var(--muted);
        font-size: 12px;
      }}
      .empty {{
        border: 1px dashed rgba(231, 237, 247, 0.25);
        border-radius: 16px;
        padding: 16px;
        color: var(--muted);
        background: rgba(15, 23, 38, 0.4);
      }}
      @media (max-width: 980px) {{
        .layout {{
          grid-template-columns: 1fr;
          grid-template-rows: 220px minmax(0, 1fr);
        }}
        aside {{
          border-right: none;
          border-bottom: 1px solid var(--border);
        }}
        .grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
  </head>
  <body>
    <header>
      <div>
        <h1>{safe_title}</h1>
        <div class="meta">Run: {escape(job_id)}</div>
      </div>
      <div class="meta">Generated by RAVEN WebUI</div>
    </header>

    <div class="layout">
      <aside>
        <div class="search">
          <input id="q" type="search" placeholder="Filter plots…" />
        </div>
        <div id="list" class="list"></div>
      </aside>
      <main>
        <div id="grid" class="grid"></div>
        <div id="empty" class="empty" style="display:none;">
          No plot assets found yet. Run may still be executing, or it did not produce plots.
        </div>
      </main>
    </div>

    <script>
      const JOB_ID = {escape(repr(job_id))};
      let ITEMS = [{encoded_items}];

      function fileUrl(path) {{
        const params = new URLSearchParams({{ path }});
        return `/api/xml-builder/run/${{encodeURIComponent(JOB_ID)}}/file?${{params.toString()}}`;
      }}

      function normalize(s) {{
        return String(s || "").toLowerCase();
      }}

      function isPlotAsset(it) {{
        if (!it) return false;
        if (it.kind === "html") return true;
        if (it.kind === "image") return true;
        if (it.kind === "pdf") return true;
        return false;
      }}

      function render(filter) {{
        const q = normalize(filter);
        const list = document.getElementById("list");
        const grid = document.getElementById("grid");
        const empty = document.getElementById("empty");
        list.innerHTML = "";
        grid.innerHTML = "";

        const filtered = ITEMS
          .filter(isPlotAsset)
          .filter((it) => normalize(it.label).includes(q) || normalize(it.path).includes(q));
        if (filtered.length === 0) {{
          empty.style.display = ITEMS.length === 0 ? "block" : "block";
          return;
        }}
        empty.style.display = "none";

        for (const it of filtered) {{
          const btn = document.createElement("button");
          btn.type = "button";
          btn.textContent = it.label;
          btn.title = it.path;
          btn.addEventListener("click", () => {{
            const el = document.getElementById(`card-${{it.path}}`);
            if (el) el.scrollIntoView({{ behavior: "smooth", block: "start" }});
          }});
          list.appendChild(btn);

          const card = document.createElement("section");
          card.className = "card";
          card.id = `card-${{it.path}}`;
          const header = document.createElement("div");
          header.className = "card__header";
          const title = document.createElement("div");
          title.className = "card__title";
          title.textContent = it.label;
          const actions = document.createElement("div");
          actions.className = "card__actions";
          const a = document.createElement("a");
          a.href = fileUrl(it.path);
          a.target = "_blank";
          a.rel = "noreferrer";
          a.textContent = "Open";
          actions.appendChild(a);
          header.appendChild(title);
          header.appendChild(actions);
          card.appendChild(header);
          const kind = it.kind || "file";
          if (kind === "html") {{
            const iframe = document.createElement("iframe");
            iframe.loading = "lazy";
            iframe.src = fileUrl(it.path);
            iframe.sandbox = "allow-scripts allow-same-origin allow-forms allow-popups";
            card.appendChild(iframe);
          }} else if (kind === "image") {{
            const img = document.createElement("img");
            img.className = "asset";
            img.loading = "lazy";
            img.src = fileUrl(it.path);
            img.alt = it.label;
            card.appendChild(img);
          }} else if (kind === "pdf") {{
            const iframe = document.createElement("iframe");
            iframe.loading = "lazy";
            iframe.src = fileUrl(it.path);
            card.appendChild(iframe);
          }} else {{
            const preview = document.createElement("div");
            preview.className = "file-preview";
            const size = typeof it.size === "number" && it.size > 0 ? `${{it.size}} bytes` : "Size unknown";
            preview.innerHTML = `<div>File type: ${{kind}}</div><div>${{size}}</div>`;
            card.appendChild(preview);
          }}
          grid.appendChild(card);
        }}
      }}

      const q = document.getElementById("q");
      q.addEventListener("input", () => render(q.value));
      async function refreshItems() {{
        try {{
          const resp = await fetch(`/api/xml-builder/run/${{encodeURIComponent(JOB_ID)}}/outputs?limit=400`);
          if (!resp.ok) {{
            render(q.value);
            return;
          }}
          const payload = await resp.json();
          const items = Array.isArray(payload.items) ? payload.items : [];
          if (items.length) {{
            ITEMS = items.filter((it) => it && typeof it.path === "string").map((it) => {{
              return {{
                path: it.path,
                label: it.label || it.path,
                kind: it.kind || "file",
                size: typeof it.size === "number" ? it.size : 0,
              }};
            }});
          }}
          render(q.value);
        }} catch (_err) {{
          render(q.value);
        }}
      }}
      // Initial render from embedded snapshot, then refresh from server.
      render("");
      refreshItems();
      // While the run may still be producing plots, refresh a few times.
      let refreshCount = 0;
      const timer = setInterval(() => {{
        refreshCount += 1;
        refreshItems();
        if (refreshCount >= 10) clearInterval(timer);
      }}, 2000);
    </script>
  </body>
</html>
"""
