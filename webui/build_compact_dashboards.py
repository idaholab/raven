"""
Generate compact HTML dashboards for one or more RAVEN run directories.

Usage:
  python -m webui.build_compact_dashboards --run-dir /path/to/run
  python -m webui.build_compact_dashboards --runs-root /path/to/webui_runs
"""

from __future__ import annotations

import argparse
import ast
import datetime
import html
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _default_runs_root() -> Path:
  return Path(__file__).resolve().parents[1] / "webui_runs"


def _iter_run_dirs(run_dir: Path, runs_root: Path) -> Iterable[Path]:
  if run_dir is not None:
    yield run_dir
    return
  for child in sorted(runs_root.iterdir()):
    if not child.is_dir():
      continue
    if child.name.startswith("_"):
      continue
    yield child


EXT_ORDER = [".gif", ".html", ".png"]

AXES = [
  ("core", "Core Layouts (BOC/EOC)"),
  ("surface", "3D Surfaces / Bars"),
  ("anim", "Animated / Time Series"),
  ("opt2d", "Optimization 2D"),
  ("radial", "Radial / Circular"),
  ("templates", "Templates / Symmetry"),
]


def _axis_key(stem: str) -> str:
  key = stem.lower()
  if "surface" in key or "bars" in key:
    return "surface"
  if "template" in key or "final" in key or "quarter" in key or "eighth" in key:
    return "templates"
  if key.startswith("core_") or "core_" in key:
    return "core"
  if any(
    item in key
    for item in (
      "front_evolution",
      "front_rank",
      "hypervolume",
      "animation",
      "objective_contour",
      "preference_sweep",
    )
  ):
    return "anim"
  if any(item in key for item in ("radviz", "chord", "glyph", "star_coordinates")):
    return "radial"
  return "opt2d"


def _humanize(name: str) -> str:
  clean = name.replace("_", " ").strip()
  return " ".join(
    token.upper() if token in {"boc", "eoc", "rpf", "nsga"} else token.title()
    for token in clean.split()
  )


def _collect_assets(run_dir: Path) -> List[Dict]:
  grouped: Dict[str, Dict] = {}
  for path in run_dir.iterdir():
    if path.suffix.lower() not in EXT_ORDER:
      continue
    stem = path.stem
    entry = grouped.setdefault(
      stem,
      {
        "id": stem,
        "stem": stem,
        "axis": _axis_key(stem),
        "files": {},
      },
    )
    entry["files"][path.suffix.lower()] = path.name

  ordered: List[Dict] = []
  for stem, entry in grouped.items():
    primary_ext = next((ext for ext in EXT_ORDER if ext in entry["files"]), None)
    if primary_ext is None:
      continue
    entry["primary_ext"] = primary_ext
    entry["primary_file"] = entry["files"][primary_ext]
    entry["label"] = _humanize(stem)
    entry["variants"] = [f for ext, f in entry["files"].items() if f != entry["primary_file"]]
    ordered.append(entry)

  axis_order = [key for key, _ in AXES]
  return sorted(
    ordered,
    key=lambda item: (
      axis_order.index(item["axis"]) if item["axis"] in axis_order else len(axis_order),
      item["label"],
    ),
  )


def _detect_objective_count(run_dir: Path) -> int:
  xml_candidates: List[Path] = []
  input_xml = run_dir / "input.xml"
  if input_xml.exists():
    xml_candidates.append(input_xml)
  xml_candidates.extend(sorted(run_dir.glob(".webui_input_*.xml")))

  for xml_path in xml_candidates:
    try:
      text = xml_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
      continue
    counts: List[int] = []
    for tag in ("objective", "objectives"):
      for match in re.findall(rf"<{tag}>([^<]+)</{tag}>", text, flags=re.IGNORECASE):
        tokens = [item.strip() for item in match.split(",") if item.strip()]
        if tokens:
          counts.append(len(tokens))
    for match in re.findall(r"<type>([^<]+)</type>", text, flags=re.IGNORECASE):
      tokens = [item.strip() for item in match.split(",") if item.strip()]
      if not tokens:
        continue
      lowered = [item.lower() for item in tokens]
      if all(item in {"min", "max", "maximize", "minimize"} for item in lowered):
        counts.append(len(tokens))
    if counts:
      return max(counts)

  csv_candidates = sorted(run_dir.glob("opt_export*.csv"))
  if not csv_candidates:
    csv_candidates = sorted(run_dir.glob("*.csv"))
  for csv_path in csv_candidates:
    try:
      header = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
      continue
    fields = [item.strip() for item in header.split(",") if item.strip()]
    fitness_cols = [item for item in fields if item.startswith("FitnessEvaluation_")]
    if fitness_cols:
      return len(fitness_cols)
    obj_cols = [item for item in fields if re.match(r"obj\d+$", item, flags=re.IGNORECASE)]
    if obj_cols:
      return len(obj_cols)

  return 2


def _normalize_plot_stem(stem: str) -> str:
  base = re.sub(r"^\d+-", "", stem or "")
  base = re.sub(r"_frames_\d+$", "", base)
  base = re.sub(r"_rank_animation$", "", base)
  base = re.sub(r"_\d+(?:\.\d+)?$", "", base)
  return base


def _strip_namespace(tag: str) -> str:
  return tag.split("}", 1)[-1] if "}" in tag else tag


def _parse_plot_definitions(run_dir: Path, job_id: Optional[str] = None) -> Dict[str, Dict[str, str]]:
  xml_candidates: List[Path] = []
  input_xml = run_dir / "input.xml"
  if input_xml.exists():
    xml_candidates.append(input_xml)
  xml_candidates.extend(sorted(run_dir.glob(".webui_input_*.xml")))
  if job_id:
    jobs_root = Path(__file__).resolve().parents[1] / "webui_runs" / "_jobs" / job_id
    job_input = jobs_root / "input.xml"
    if job_input.exists():
      xml_candidates.append(job_input)

  plots: Dict[str, Dict[str, str]] = {}
  for xml_path in xml_candidates:
    try:
      root = ET.fromstring(xml_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
      continue
    for plot in root.findall(".//Plot"):
      name = plot.get("name") or ""
      subtype = plot.get("subType") or plot.get("subtype") or ""
      if not name:
        continue
      params: Dict[str, str] = {}
      for child in list(plot):
        tag = _strip_namespace(child.tag)
        text = (child.text or "").strip()
        if text:
          params[tag] = text
      plots[name] = {"name": name, "subtype": subtype, **params}
  return plots


def _first_paragraph(text: str) -> str:
  chunks = [chunk.strip() for chunk in re.split(r"\\n\\s*\\n", text.strip()) if chunk.strip()]
  return chunks[0] if chunks else ""


def _load_plot_docstrings() -> Dict[str, str]:
  plot_dir = Path(__file__).resolve().parents[1] / "ravenframework" / "OutStreams" / "PlotInterfaces"
  docstrings: Dict[str, str] = {}
  if not plot_dir.exists():
    return docstrings
  for path in sorted(plot_dir.glob("*.py")):
    if path.name in {"__init__.py", "Factory.py", "PlotInterface.py"}:
      continue
    try:
      module = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
      continue
    try:
      tree = ast.parse(module)
    except Exception:
      continue
    module_doc = ast.get_docstring(tree)
    if module_doc:
      docstrings[path.stem] = _first_paragraph(module_doc)
    for node in getattr(tree, "body", []):
      if not isinstance(node, ast.ClassDef):
        continue
      name = node.name
      doc = ast.get_docstring(node)
      if doc:
        docstrings[name] = _first_paragraph(doc)
  return docstrings


def _clean_manual_text(text: str) -> str:
  text = re.sub(r"\\\\begin\\{lstlisting\\}.*?\\\\end\\{lstlisting\\}", " ", text, flags=re.DOTALL)
  text = re.sub(r"\\\\xml(?:String|Attr|Node|Desc)\\{([^}]+)\\}", r"\\1", text)
  text = re.sub(r"\\\\textbf\\{([^}]+)\\}", r"\\1", text)
  text = re.sub(r"\\\\ref\\{[^}]+\\}", "", text)
  text = re.sub(r"\\\\cite\\{[^}]+\\}", "", text)
  text = re.sub(r"\\\\item", "-", text)
  text = re.sub(r"\\\\[a-zA-Z]+", "", text)
  text = re.sub(r"[{}]", "", text)
  text = re.sub(r"\\s+", " ", text).strip()
  return text


def _load_plot_manual() -> Dict[str, str]:
  manual_path = Path(__file__).resolve().parents[1] / "doc" / "user_manual" / "OutStreamSystem.tex"
  if not manual_path.exists():
    return {}
  try:
    text = manual_path.read_text(encoding="utf-8", errors="ignore")
  except OSError:
    return {}
  blocks = re.findall(r"\\\\subsubsection\\{([^}]+)\\}(.*?)(?=\\\\subsubsection\\{|\\\\subsection\\{|\\\\section\\{|\\Z)", text, flags=re.DOTALL)
  manual: Dict[str, str] = {}
  for title, body in blocks:
    cleaned = _clean_manual_text(body)
    if cleaned:
      manual[title.strip()] = _first_paragraph(cleaned)
  return manual


def _format_plot_config(meta: Dict[str, str]) -> str:
  keys = ["source", "vars", "objectives", "objective", "axes", "index", "format", "how"]
  parts = []
  for key in keys:
    if key in meta:
      parts.append(f"{key}: {meta[key]}")
  return " | ".join(parts)


def _annotate_assets(run_dir: Path, assets: List[Dict], job_id: Optional[str] = None) -> None:
  plot_defs = _parse_plot_definitions(run_dir, job_id=job_id)
  manual = _load_plot_manual()
  docstrings = _load_plot_docstrings()
  for asset in assets:
    stem = asset.get("stem", asset.get("id", ""))
    plot_name = _normalize_plot_stem(stem)
    meta = plot_defs.get(plot_name, {})
    subtype = meta.get("subtype", "")
    manual_text = manual.get(subtype, "")
    doc_text = docstrings.get(subtype, "") or docstrings.get(subtype.replace("-", ""), "")
    parts = []
    if manual_text:
      parts.append(manual_text)
    if doc_text and doc_text not in manual_text:
      parts.append(doc_text)
    config = _format_plot_config(meta)
    if config:
      parts.append(f"Run config: {config}")
    description = "\n\n".join(parts) if parts else "No description available."
    asset["plot_name"] = plot_name or asset.get("id", "")
    asset["subtype"] = subtype
    asset["description"] = description


def _render_html(
  title: str,
  run_dir: Path,
  assets: List[Dict],
  job_id: Optional[str],
  objective_count: int,
  file_base: Optional[str],
) -> str:
  timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
  data_json = json.dumps(assets, indent=2)
  axes_json = json.dumps([{"key": key, "label": label} for key, label in AXES])
  safe_job_id = html.escape(job_id) if job_id else ""
  safe_file_base = html.escape(file_base) if file_base else ""
  subplot_span = max(2, min(3, objective_count))
  return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    :root {{
      --bg: #04060f;
      --panel: #0a1020;
      --accent: #4ff0f5;
      --accent-2: #ff5fd7;
      --text: #e7edff;
      --muted: #8ea5c7;
      --glow: 0 0 14px rgba(79, 240, 245, 0.55);
      --radius: 16px;
    }}
    [data-theme="light"] {{
      --bg: #f6f7fb;
      --panel: #ffffff;
      --accent: #0aa0ff;
      --accent-2: #ff3f9e;
      --text: #0b1324;
      --muted: #4a5b7a;
      --glow: 0 0 14px rgba(10, 160, 255, 0.35);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 0;
      font-family: 'Space Grotesk', 'JetBrains Mono', 'Menlo', monospace;
      background: radial-gradient(circle at 15% 20%, rgba(255, 95, 215, 0.12), transparent 30%),
                  radial-gradient(circle at 80% 15%, rgba(79, 240, 245, 0.12), transparent 28%),
                  radial-gradient(circle at 45% 75%, rgba(255, 178, 64, 0.12), transparent 30%),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
    }}
    [data-theme="light"] body {{
      background: radial-gradient(circle at 10% 15%, rgba(10, 160, 255, 0.15), transparent 30%),
                  radial-gradient(circle at 80% 20%, rgba(255, 63, 158, 0.12), transparent 28%),
                  radial-gradient(circle at 50% 85%, rgba(255, 180, 75, 0.12), transparent 30%),
                  var(--bg);
    }}
    header {{
      padding: 24px 20px 10px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px 20px;
      align-items: baseline;
    }}
    h1 {{
      margin: 0;
      font-size: 26px;
      letter-spacing: 0.7px;
      text-transform: uppercase;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }}
    .pill {{
      padding: 6px 10px;
      border-radius: 999px;
      background: linear-gradient(90deg, rgba(79, 240, 245, 0.22), rgba(255, 95, 215, 0.18));
      color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.08);
      box-shadow: var(--glow);
      font-weight: 600;
    }}
    .theme-toggle {{
      border: 1px solid rgba(255,255,255,0.18);
      background: rgba(255,255,255,0.08);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 999px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .theme-toggle:hover {{
      border-color: var(--accent);
      box-shadow: var(--glow);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      gap: 16px;
      padding: 0 20px 24px;
    }}
    .sidebar {{
      background: rgba(10, 16, 32, 0.85);
      border-radius: 14px;
      border: 1px solid rgba(255,255,255,0.08);
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      min-height: 420px;
    }}
    [data-theme="light"] .sidebar {{
      background: rgba(255, 255, 255, 0.9);
    }}
    .sidebar h2 {{
      margin: 6px 4px 2px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }}
    .search input {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,0.14);
      background: rgba(255,255,255,0.06);
      padding: 8px 10px;
      color: var(--text);
    }}
    .asset-list {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      overflow: auto;
      max-height: 420px;
    }}
    .asset-item {{
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 10px;
      cursor: pointer;
      font-size: 12px;
      text-align: left;
    }}
    .asset-item:hover {{
      border-color: rgba(79, 240, 245, 0.35);
      box-shadow: var(--glow);
    }}
    .asset-item.disabled,
    .asset-item:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
      box-shadow: none;
      border-color: rgba(255,255,255,0.08);
    }}
    .description {{
      margin-top: 8px;
      padding: 10px;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.12);
      background: rgba(255,255,255,0.04);
      font-size: 12px;
      color: var(--text);
      line-height: 1.4;
      max-height: 220px;
      overflow: auto;
      white-space: pre-wrap;
    }}
    .description .desc-title {{
      font-weight: 700;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 11px;
      color: var(--muted);
    }}
    .add-tile {{
      margin-top: 6px;
      background: rgba(79, 240, 245, 0.14);
      border: 1px solid rgba(79, 240, 245, 0.5);
      color: var(--text);
      padding: 8px 10px;
      border-radius: 10px;
      font-weight: 600;
      cursor: pointer;
    }}
    .deck {{
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      align-content: start;
    }}
    .row-break {{
      grid-column: 1 / -1;
      height: 0;
    }}
    .card {{
      background: var(--panel);
      border-radius: var(--radius);
      border: 1px solid rgba(255,255,255,0.08);
      box-shadow: 0 12px 30px rgba(0,0,0,0.35);
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr;
      min-height: 320px;
      transition: transform 0.2s ease, border-color 0.2s ease;
      position: relative;
    }}
    .card:hover {{
      transform: translateY(-3px);
      border-color: rgba(79, 240, 245, 0.35);
      box-shadow: 0 18px 44px rgba(0,0,0,0.45), var(--glow);
    }}
    .card.selected {{
      border-color: rgba(255, 95, 215, 0.5);
      box-shadow: 0 0 0 2px rgba(255, 95, 215, 0.25);
    }}
    .remove-btn {{
      position: absolute;
      top: 8px;
      right: 8px;
      width: 22px;
      height: 22px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.18);
      background: rgba(255,255,255,0.08);
      color: var(--text);
      font-weight: 700;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      z-index: 3;
    }}
    .remove-btn:hover {{
      border-color: var(--accent-2);
      color: var(--accent-2);
      box-shadow: var(--glow);
    }}
    .card-head {{
      padding: 12px 14px;
      display: grid;
      gap: 10px;
      background: linear-gradient(120deg, rgba(79,240,245,0.12), rgba(255,95,215,0.10));
      border-bottom: 1px solid rgba(255,255,255,0.06);
    }}
    .card-head .title {{
      font-weight: 700;
      letter-spacing: 0.3px;
      text-transform: uppercase;
    }}
    .controls {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .open-link {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 10px;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      color: var(--text);
      text-decoration: none;
      font-weight: 600;
      font-size: 12px;
    }}
    .open-link:hover {{
      border-color: var(--accent);
      color: var(--accent);
      box-shadow: var(--glow);
    }}
    .body {{
      padding: 12px;
      display: grid;
      gap: 10px;
      grid-template-rows: 1fr auto auto;
    }}
    .viewport {{
      position: relative;
      background: #060914;
      border-radius: 12px;
      border: 1px solid rgba(255,255,255,0.08);
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      height: 240px;
    }}
    .viewport img, .viewport iframe, .viewport canvas {{
      width: 100%;
      height: 100%;
      border: none;
      object-fit: contain;
      background: #0c0f1b;
    }}
    .toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .btn {{
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.12);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 9px;
      cursor: pointer;
      font-weight: 600;
      font-size: 12px;
    }}
    .radio-group {{
      display: flex;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }}
    .chip {{
      padding: 6px 9px;
      border-radius: 9px;
      background: rgba(255,255,255,0.06);
      border: 1px dashed rgba(255,255,255,0.12);
      color: var(--text);
      text-decoration: none;
      font-size: 12px;
    }}
    .variants {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .variant {{
      padding: 6px 9px;
      border-radius: 9px;
      background: rgba(255,255,255,0.06);
      border: 1px dashed rgba(255,255,255,0.12);
      color: var(--text);
      text-decoration: none;
      font-size: 12px;
    }}
    .variant:hover {{
      border-color: var(--accent);
      color: var(--accent);
      box-shadow: var(--glow);
    }}
    .resize-handle {{
      position: absolute;
      right: 6px;
      bottom: 6px;
      width: 16px;
      height: 16px;
      cursor: nwse-resize;
      border-right: 2px solid rgba(255,255,255,0.35);
      border-bottom: 2px solid rgba(255,255,255,0.35);
      opacity: 0.6;
      z-index: 2;
    }}
    .resize-handle:hover {{
      opacity: 1;
    }}
    @media (max-width: 1200px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .deck {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 720px) {{
      .deck {{ grid-template-columns: 1fr; }}
    }}
  </style>
  <script src="https://unpkg.com/gifuct-js@2.1.2/dist/gifuct.min.js"></script>
</head>
<body>
  <header>
    <div>
      <h1>{html.escape(title)}</h1>
      <div class="meta">
        <span class="pill">Run folder: {html.escape(run_dir.name)}</span>
        <span class="pill">Generated: {timestamp}</span>
        <span class="pill">Assets: {len(assets)}</span>
        <button class="theme-toggle" id="theme-toggle" type="button">Theme</button>
      </div>
    </div>
  </header>
  <div class="layout">
    <aside class="sidebar">
      <h2>Plots</h2>
      <div class="search">
        <input id="asset-filter" type="search" placeholder="Filter assets..." />
      </div>
      <div class="asset-list" id="asset-list"></div>
      <button class="add-tile" id="add-tile" type="button">Add tile</button>
      <div class="description">
        <div class="desc-title">Plot Details</div>
        <div id="plot-description">Select a plot tile to see details.</div>
      </div>
    </aside>
    <main>
      <div class="deck" id="deck"></div>
    </main>
  </div>
  <script>
    const JOB_ID = {json.dumps(safe_job_id)};
    const FILE_BASE = {json.dumps(safe_file_base)};
    const assets = {data_json};
    const allAxes = {axes_json};
    const axisKeys = new Set(assets.map(a => a.axis));
    const axes = allAxes.filter(a => axisKeys.has(a.key));
    const players = new Map();
    const fileBase = FILE_BASE || (JOB_ID ? `/api/xml-builder/run/${{encodeURIComponent(JOB_ID)}}/file` : '');
    const objectiveCount = {objective_count};
    const defaultSubplotSpan = {subplot_span};
    let activeTile = null;
    const themeKey = 'raven-dashboard-theme';

    function setTheme(theme) {{
      document.documentElement.setAttribute('data-theme', theme);
      try {{
        localStorage.setItem(themeKey, theme);
      }} catch (_err) {{
        // ignore storage issues
      }}
    }}

    function themeIconSvg(kind) {{
      if (kind === 'dark') {{
        return '<svg width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"><circle cx=\"12\" cy=\"12\" r=\"5\" stroke=\"currentColor\" stroke-width=\"2\"/><path d=\"M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linecap=\"round\"/></svg>';
      }}
      return '<svg width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\"><path d=\"M21 14.5A8.5 8.5 0 1 1 9.5 3a7 7 0 0 0 11.5 11.5Z\" stroke=\"currentColor\" stroke-width=\"2\" stroke-linejoin=\"round\"/></svg>';
    }}

    function initTheme() {{
      let theme = 'dark';
      try {{
        theme = localStorage.getItem(themeKey) || theme;
      }} catch (_err) {{
        // ignore storage issues
      }}
      setTheme(theme);
      const toggle = document.getElementById('theme-toggle');
      if (toggle) {{
        toggle.innerHTML = themeIconSvg(theme) + (theme === 'light' ? 'Dark' : 'Light');
        toggle.addEventListener('click', () => {{
          const next = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
          setTheme(next);
          toggle.innerHTML = themeIconSvg(next) + (next === 'light' ? 'Dark' : 'Light');
        }});
      }}
    }}

    function assetUrl(filename) {{
      if (!fileBase) return filename;
      const url = new URL(fileBase, window.location.origin);
      url.searchParams.set('path', filename);
      return url.toString();
    }}

    function optionsForAxis(key) {{
      return assets.filter(asset => asset.axis === key);
    }}

    function waitForGifuct(timeoutMs = 2000) {{
      if (window.gifuct) return Promise.resolve(window.gifuct);
      return new Promise((resolve, reject) => {{
        const start = Date.now();
        const timer = setInterval(() => {{
          if (window.gifuct) {{
            clearInterval(timer);
            resolve(window.gifuct);
            return;
          }}
          if (Date.now() - start > timeoutMs) {{
            clearInterval(timer);
            reject(new Error('gifuct not available'));
          }}
        }}, 50);
      }});
    }}

    function escapeHtml(text) {{
      return String(text || '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\"/g, '&quot;')
        .replace(/'/g, '&#39;');
    }}

    function updateDescription(assetId) {{
      const panel = document.getElementById('plot-description');
      if (!panel) return;
      const asset = assets.find(a => a.id === assetId);
      if (!asset) {{
        panel.textContent = 'Select a plot tile to see details.';
        return;
      }}
      const name = asset.plot_name || asset.label || asset.id;
      const subtype = asset.subtype ? ` (${{asset.subtype}})` : '';
      const description = asset.description || 'No description available.';
      panel.innerHTML = `<strong>${{escapeHtml(name + subtype)}}</strong>\\n\\n${{escapeHtml(description)}}`;
    }}

    function fitIframe(iframe) {{
      try {{
        const doc = iframe.contentDocument;
        if (!doc) return;
        const content = doc.documentElement;
        const container = iframe.parentElement;
        if (!content || !container) return;
        const rect = container.getBoundingClientRect();
        const contentWidth = Math.max(content.scrollWidth, content.offsetWidth);
        const contentHeight = Math.max(content.scrollHeight, content.offsetHeight);
        if (!contentWidth || !contentHeight || !rect.width || !rect.height) return;
        const scale = Math.min(rect.width / contentWidth, rect.height / contentHeight, 1);
        iframe.style.transformOrigin = '0 0';
        iframe.style.transform = `scale(${{scale}})`;
        iframe.style.width = `${{contentWidth}}px`;
        iframe.style.height = `${{contentHeight}}px`;
      }} catch (_err) {{
        // Ignore cross-origin or sizing issues.
      }}
    }}

    function renderVariants(tileId, asset) {{
      const wrap = document.getElementById('variants-' + tileId);
      if (!wrap) return;
      wrap.innerHTML = '';
      if (asset.variants && asset.variants.length) {{
        asset.variants.forEach(v => {{
          const a = document.createElement('a');
          a.className = 'variant';
          a.href = assetUrl(v);
          a.target = '_blank';
          a.textContent = v.split('.').pop().toUpperCase();
          wrap.appendChild(a);
        }});
      }}
    }}

    function clearPlayer(tileId) {{
      const player = players.get(tileId);
      if (player && player.destroy) player.destroy();
      players.delete(tileId);
    }}

    function buildGifPlayer(tileId, src, canvas) {{
      const ctx = canvas.getContext('2d');
      let frames = [];
      let idx = 0;
      let mode = 'loop';
      let playing = true;
      let fast = false;
      let rafId = null;
      let nextTime = 0;

      function schedule() {{
        if (!playing || frames.length === 0) return;
        const frame = frames[idx];
        const delay = Math.max(20, (frame.delay || 8) * 10) / (fast ? 2 : 1);
        const now = performance.now();
        if (nextTime === 0) nextTime = now;
        const wait = Math.max(0, nextTime - now);
        rafId = setTimeout(() => {{
          drawFrame(frame);
          advance();
          nextTime = performance.now() + delay;
          schedule();
        }}, wait);
      }}

      function advance() {{
        idx += 1;
        if (idx >= frames.length) {{
          if (mode === 'loop') idx = 0;
          else if (mode === 'reflect') {{
            frames.reverse();
            idx = 1;
          }} else {{
            playing = false;
          }}
        }}
      }}

      function drawFrame(frame) {{
        if (!frame) return;
        canvas.width = frame.dims.width;
        canvas.height = frame.dims.height;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const imageData = ctx.createImageData(frame.dims.width, frame.dims.height);
        imageData.data.set(frame.patch);
        ctx.putImageData(imageData, frame.dims.left, frame.dims.top);
      }}

      fetch(src)
        .then(r => r.arrayBuffer())
        .then(buf => {{
          if (!window.gifuct) throw new Error('gifuct missing (offline/CDN blocked)');
          const gif = window.gifuct.parseGIF(buf);
          frames = window.gifuct.decompressFrames(gif, true);
          idx = 0;
          nextTime = 0;
          playing = true;
          schedule();
        }})
        .catch(() => {{
          ctx.fillStyle = '#ff5fd7';
          ctx.font = '14px JetBrains Mono, monospace';
          ctx.fillText('GIF controls unavailable. Showing static GIF.', 10, 24);
          const img = new Image();
          img.onload = () => {{
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            ctx.drawImage(img, 0, 0);
          }};
          img.src = src;
        }});

      players.set(tileId, {{
        play: () => {{ if (!playing) {{ playing = true; nextTime = 0; schedule(); }} }},
        pause: () => {{ playing = false; if (rafId) clearTimeout(rafId); }},
        rewind: () => {{ idx = 0; nextTime = 0; playing = true; schedule(); }},
        toggleFast: () => {{ fast = !fast; }},
        setMode: (m) => {{ mode = m; }},
        destroy: () => {{ playing = false; if (rafId) clearTimeout(rafId); }},
      }});
    }}

    function renderAsset(tileId, assetId) {{
      clearPlayer(tileId);
      const viewport = document.getElementById('view-' + tileId);
      if (!viewport) return;
      viewport.innerHTML = '';
      const asset = assets.find(a => a.id === assetId);
      if (!asset) {{
        viewport.innerHTML = '<div class="chip">Asset not found</div>';
        return;
      }}
      renderVariants(tileId, asset);
      const toolbar = document.getElementById('toolbar-' + tileId);
      const openLink = document.getElementById('open-' + tileId);
      const gifRequested = asset.primary_ext === '.gif' && location.protocol !== 'file:';
      const gifControls = gifRequested && window.gifuct;
      if (toolbar) {{
        toolbar.style.display = gifRequested ? 'flex' : 'none';
        if (gifControls) {{
          const loopInput = toolbar.querySelector('input[value=\"loop\"]');
          if (loopInput) loopInput.checked = true;
        }}
      }}
      if (openLink) {{
        if (asset.primary_ext === '.html') {{
          openLink.href = assetUrl(asset.primary_file);
          openLink.style.display = 'inline-flex';
        }} else {{
          openLink.style.display = 'none';
        }}
      }}
      if (asset.primary_ext === '.html') {{
        const iframe = document.createElement('iframe');
        iframe.src = assetUrl(asset.primary_file);
        iframe.loading = 'lazy';
        iframe.addEventListener('load', () => {{
          fitIframe(iframe);
          if (window.ResizeObserver) {{
            const observer = new ResizeObserver(() => fitIframe(iframe));
            observer.observe(viewport);
          }}
        }});
        viewport.appendChild(iframe);
        return;
      }}
      if (asset.primary_ext === '.gif') {{
        const canvas = document.createElement('canvas');
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        viewport.appendChild(canvas);
        const src = assetUrl(asset.primary_file);
        waitForGifuct()
          .then(() => {{
            buildGifPlayer(tileId, src, canvas);
            const player = players.get(tileId);
            if (player) {{
              player.setMode('loop');
              player.play();
            }}
          }})
          .catch(() => {{
            viewport.innerHTML = '';
            const img = document.createElement('img');
            img.src = src;
            img.loading = 'lazy';
            viewport.appendChild(img);
          }});
        return;
      }}
      const img = document.createElement('img');
      img.src = assetUrl(asset.primary_file);
      img.loading = 'lazy';
      viewport.appendChild(img);
    }}

    function setActiveTile(tile) {{
      if (activeTile) activeTile.classList.remove('selected');
      activeTile = tile;
      if (activeTile) activeTile.classList.add('selected');
      if (activeTile && activeTile.dataset.asset) {{
        updateDescription(activeTile.dataset.asset);
      }}
    }}

    function setTileAsset(tile, assetId) {{
      tile.dataset.asset = assetId;
      const asset = assets.find(a => a.id === assetId);
      const title = tile.querySelector('.title');
      if (title) title.textContent = asset ? asset.label : 'Empty';
      renderAsset(tile.id, assetId);
      updateAssetListState();
      if (activeTile === tile) {{
        updateDescription(assetId);
      }}
    }}

    function swapTileAssets(tileA, tileB) {{
      const assetA = tileA.dataset.asset;
      const assetB = tileB.dataset.asset;
      if (assetA) setTileAsset(tileB, assetA);
      if (assetB) setTileAsset(tileA, assetB);
      updateAssetListState();
    }}

    function swapTileOrder(source, target) {{
      if (!source || !target) return;
      const parent = source.parentNode;
      if (!parent || parent !== target.parentNode) return;
      const sourceNext = source.nextSibling === target ? source : source.nextSibling;
      parent.insertBefore(source, target);
      if (sourceNext) {{
        parent.insertBefore(target, sourceNext);
      }} else {{
        parent.appendChild(target);
      }}
    }}

    function beginResize(tile, startX) {{
      const deck = document.getElementById('deck');
      if (!deck) return;
      const deckRect = deck.getBoundingClientRect();
      const gap = parseFloat(getComputedStyle(deck).columnGap || '0');
      const colWidth = (deckRect.width - gap * 3) / 4;

      function onMove(event) {{
        const tileRect = tile.getBoundingClientRect();
        const desiredWidth = Math.max(1, event.clientX - tileRect.left);
        const span = Math.max(1, Math.min(4, Math.round(desiredWidth / (colWidth + gap))));
        tile.dataset.span = String(span);
        tile.style.gridColumn = `span ${{span}}`;
      }}
      function onUp() {{
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
      }}
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    }}

    function createTile(slot) {{
      const card = document.createElement('div');
      card.className = 'card';
      card.id = slot.id;
      card.dataset.span = String(slot.span);
      card.dataset.role = slot.role || '';
      card.draggable = true;
      card.style.gridColumn = `span ${{slot.span}}`;

      const removeBtn = document.createElement('button');
      removeBtn.className = 'remove-btn';
      removeBtn.type = 'button';
      removeBtn.title = 'Remove tile';
      removeBtn.textContent = '×';
      removeBtn.addEventListener('click', (event) => {{
        event.stopPropagation();
        card.remove();
        if (activeTile === card) {{
          activeTile = null;
          updateDescription('');
        }}
        updateAssetListState();
      }});
      card.appendChild(removeBtn);

      const head = document.createElement('div');
      head.className = 'card-head';
      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = slot.label;
      head.appendChild(title);

      const selectWrap = document.createElement('div');
      selectWrap.className = 'controls';
      if (slot.role === 'subplot') {{
        const sel = document.createElement('select');
        sel.dataset.axis = 'subplot';
        selectWrap.appendChild(sel);
      }}
      const openLink = document.createElement('a');
      openLink.className = 'open-link';
      openLink.id = 'open-' + slot.id;
      openLink.target = '_blank';
      openLink.rel = 'noreferrer';
      openLink.textContent = 'Open HTML';
      openLink.style.display = 'none';
      selectWrap.appendChild(openLink);
      head.appendChild(selectWrap);
      card.appendChild(head);

      const body = document.createElement('div');
      body.className = 'body';
      const viewport = document.createElement('div');
      viewport.className = 'viewport';
      viewport.id = 'view-' + slot.id;
      body.appendChild(viewport);

      const toolbar = document.createElement('div');
      toolbar.className = 'toolbar';
      toolbar.id = 'toolbar-' + slot.id;
      toolbar.style.display = 'none';
      toolbar.innerHTML = `
        <button class="btn" data-action="play" type="button">Play</button>
        <button class="btn" data-action="pause" type="button">Pause</button>
        <button class="btn" data-action="rewind" type="button">Rewind</button>
        <button class="btn" data-action="ff" type="button">Fast</button>
        <div class="radio-group">
          <label><input type="radio" name="mode-${{slot.id}}" value="loop" checked> Loop</label>
          <label><input type="radio" name="mode-${{slot.id}}" value="once"> Once</label>
          <label><input type="radio" name="mode-${{slot.id}}" value="reflect"> Reflect</label>
        </div>
        <div class="chip" id="chip-${{slot.id}}">Animations: GIFs only</div>
      `;
      body.appendChild(toolbar);

      const variants = document.createElement('div');
      variants.className = 'variants';
      variants.id = 'variants-' + slot.id;
      body.appendChild(variants);

      card.appendChild(body);

      if (slot.role === 'subplot') {{
        const sel = selectWrap.querySelector('select');
        const options = assets.filter(asset => {{
          const key = (asset.id || '').toLowerCase();
          return key.includes('glyph') || key.includes('objective_contour');
        }});
        options.forEach((asset, index) => {{
          const option = document.createElement('option');
          option.value = asset.id;
          option.textContent = asset.label;
          if (index === 0) option.selected = true;
          sel.appendChild(option);
        }});
        sel.onchange = () => {{
          card.dataset.asset = sel.value;
          renderAsset(card.id, sel.value);
          updateAssetListState();
          if (activeTile === card) {{
            updateDescription(sel.value);
          }}
        }};
        if (options.length) {{
          card.dataset.asset = options[0].id;
          renderAsset(card.id, options[0].id);
        }}
      }}

      card.addEventListener('click', (event) => {{
        if (event.target && event.target.closest('.open-link')) return;
        setActiveTile(card);
      }});
      card.addEventListener('dragstart', (event) => {{
        event.dataTransfer.setData('text/tile', card.id);
      }});
      card.addEventListener('dragover', (event) => {{
        event.preventDefault();
      }});
      card.addEventListener('drop', (event) => {{
        event.preventDefault();
        const assetId = event.dataTransfer.getData('text/asset');
        if (assetId) {{
          setTileAsset(card, assetId);
          return;
        }}
        const sourceId = event.dataTransfer.getData('text/tile');
        if (sourceId && sourceId !== card.id) {{
          const source = document.getElementById(sourceId);
          if (source) swapTileOrder(source, card);
        }}
      }});

      const resize = document.createElement('div');
      resize.className = 'resize-handle';
      resize.addEventListener('mousedown', (event) => {{
        event.preventDefault();
        beginResize(card, event.clientX);
      }});
      card.appendChild(resize);

      return card;
    }}

    function createGroupedCard(axis) {{
      const card = document.createElement('div');
      card.className = 'card';
      card.id = `group-${{axis.key}}`;
      card.dataset.role = 'grouped';
      card.style.gridColumn = 'span 1';
      card.draggable = true;

      const head = document.createElement('div');
      head.className = 'card-head';
      const title = document.createElement('div');
      title.className = 'title';
      title.textContent = axis.label;
      head.appendChild(title);

      const selectWrap = document.createElement('div');
      selectWrap.className = 'controls';
      const sel = document.createElement('select');
      sel.dataset.axis = axis.key;
      selectWrap.appendChild(sel);
      head.appendChild(selectWrap);
      card.appendChild(head);

      const body = document.createElement('div');
      body.className = 'body';
      const viewport = document.createElement('div');
      viewport.className = 'viewport';
      viewport.id = 'view-' + card.id;
      body.appendChild(viewport);

      const toolbar = document.createElement('div');
      toolbar.className = 'toolbar';
      toolbar.id = 'toolbar-' + card.id;
      toolbar.style.display = 'none';
      toolbar.innerHTML = `
        <button class="btn" data-action="play" type="button">Play</button>
        <button class="btn" data-action="pause" type="button">Pause</button>
        <button class="btn" data-action="rewind" type="button">Rewind</button>
        <button class="btn" data-action="ff" type="button">Fast</button>
        <div class="radio-group">
          <label><input type="radio" name="mode-${{card.id}}" value="loop" checked> Loop</label>
          <label><input type="radio" name="mode-${{card.id}}" value="once"> Once</label>
          <label><input type="radio" name="mode-${{card.id}}" value="reflect"> Reflect</label>
        </div>
        <div class="chip" id="chip-${{card.id}}">Animations: GIFs only</div>
      `;
      body.appendChild(toolbar);

      const variants = document.createElement('div');
      variants.className = 'variants';
      variants.id = 'variants-' + card.id;
      body.appendChild(variants);

      card.appendChild(body);

      const removeBtn = document.createElement('button');
      removeBtn.className = 'remove-btn';
      removeBtn.type = 'button';
      removeBtn.title = 'Remove tile';
      removeBtn.textContent = '×';
      removeBtn.addEventListener('click', (event) => {{
        event.stopPropagation();
        card.remove();
        if (activeTile === card) {{
          activeTile = null;
          updateDescription('');
        }}
        updateAssetListState();
      }});
      card.appendChild(removeBtn);

      const opts = optionsForAxis(axis.key);
      opts.forEach((asset, index) => {{
        const option = document.createElement('option');
        option.value = asset.id;
        option.textContent = asset.label;
        if (index === 0) option.selected = true;
        sel.appendChild(option);
      }});

      sel.onchange = () => {{
        card.dataset.asset = sel.value;
        renderAsset(card.id, sel.value);
        updateAssetListState();
        if (activeTile === card) {{
          updateDescription(sel.value);
        }}
      }};
      toolbar.querySelectorAll('input[type=radio]').forEach(radio => {{
        radio.onchange = () => {{
          const player = players.get(card.id);
          if (player) player.setMode(radio.value);
        }};
      }});
      toolbar.querySelector('[data-action="play"]').onclick = () => {{
        const player = players.get(card.id);
        if (player) player.play();
      }};
      toolbar.querySelector('[data-action="pause"]').onclick = () => {{
        const player = players.get(card.id);
        if (player) player.pause();
      }};
      toolbar.querySelector('[data-action="rewind"]').onclick = () => {{
        const player = players.get(card.id);
        if (player) player.rewind();
      }};
      toolbar.querySelector('[data-action="ff"]').onclick = () => {{
        const player = players.get(card.id);
        if (player) player.toggleFast();
      }};

      if (opts.length) {{
        card.dataset.asset = opts[0].id;
        renderAsset(card.id, opts[0].id);
      }} else {{
        viewport.innerHTML = '<div class="chip">No assets for this axis</div>';
      }}
      card.addEventListener('click', () => {{
        setActiveTile(card);
      }});
      card.addEventListener('dragstart', (event) => {{
        event.dataTransfer.setData('text/tile', card.id);
      }});
      card.addEventListener('dragover', (event) => {{
        event.preventDefault();
      }});
      card.addEventListener('drop', (event) => {{
        event.preventDefault();
        const assetId = event.dataTransfer.getData('text/asset');
        if (assetId) {{
          card.dataset.asset = assetId;
          renderAsset(card.id, assetId);
          updateAssetListState();
          return;
        }}
        const sourceId = event.dataTransfer.getData('text/tile');
        if (sourceId && sourceId !== card.id) {{
          const source = document.getElementById(sourceId);
          if (source) swapTileOrder(source, card);
        }}
      }});
      return card;
    }}

    function collectUsedAssets() {{
      const used = new Set();
      document.querySelectorAll('.card[data-asset]').forEach(el => {{
        if (el.dataset.asset) used.add(el.dataset.asset);
      }});
      document.querySelectorAll('select[data-axis]').forEach(sel => {{
        if (sel.value) used.add(sel.value);
      }});
      return used;
    }}

    function updateAssetListState() {{
      const used = collectUsedAssets();
      document.querySelectorAll('.asset-item').forEach(btn => {{
        const assetId = btn.dataset.asset;
        const disabled = used.has(assetId);
        btn.disabled = disabled;
        btn.classList.toggle('disabled', disabled);
      }});
    }}

    function renderAssetList(filterText) {{
      const list = document.getElementById('asset-list');
      if (!list) return;
      list.innerHTML = '';
      const q = (filterText || '').toLowerCase();
      assets
        .filter(asset => asset.label.toLowerCase().includes(q) || asset.id.toLowerCase().includes(q))
        .forEach(asset => {{
          const btn = document.createElement('button');
          btn.className = 'asset-item';
          btn.type = 'button';
          btn.textContent = asset.label;
          btn.dataset.asset = asset.id;
          btn.draggable = true;
          btn.addEventListener('click', () => {{
            if (activeTile && !btn.disabled) setTileAsset(activeTile, asset.id);
          }});
          btn.addEventListener('dragstart', (event) => {{
            if (btn.disabled) return;
            event.dataTransfer.setData('text/asset', asset.id);
          }});
          list.appendChild(btn);
        }});
      updateAssetListState();
    }}

    function init() {{
      const deck = document.getElementById('deck');
      const filter = document.getElementById('asset-filter');
      const addButton = document.getElementById('add-tile');
      if (!deck) return;
      if (!assets.length) {{
        const msg = document.createElement('div');
        msg.style.padding = '18px';
        msg.style.color = '#ff47a3';
        msg.style.fontFamily = 'JetBrains Mono, monospace';
        msg.textContent = 'No plot assets (.png/.gif/.html) found in this folder.';
        deck.appendChild(msg);
        return;
      }}

      axes.forEach(axis => {{
        const card = createGroupedCard(axis);
        deck.appendChild(card);
      }});

      const breakEl = document.createElement('div');
      breakEl.className = 'row-break';
      deck.appendChild(breakEl);

      const slots = [
        {{ id: 'tile-subplot', label: 'Subplots', span: defaultSubplotSpan, role: 'subplot' }},
      ];
      slots.forEach(slot => {{
        const tile = createTile(slot);
        deck.appendChild(tile);
      }});
      setActiveTile(document.getElementById('tile-subplot'));
      renderAssetList('');
      updateAssetListState();
      if (activeTile && activeTile.dataset.asset) {{
        updateDescription(activeTile.dataset.asset);
      }}

      if (filter) {{
        filter.addEventListener('input', () => renderAssetList(filter.value));
      }}
      if (addButton) {{
        addButton.addEventListener('click', () => {{
          const id = `tile-${{Date.now()}}`;
          const tile = createTile({{ id, label: 'Custom Tile', span: 1 }});
          deck.appendChild(tile);
          setActiveTile(tile);
        }});
      }}
    }}

    initTheme();
    init();
  </script>
</body>
</html>
"""


def build_compact_dashboard_html(
  run_dir: Path,
  title: str,
  job_id: Optional[str] = None,
  file_base: Optional[str] = None,
) -> Tuple[str, int]:
  assets = _collect_assets(run_dir)
  _annotate_assets(run_dir, assets, job_id=job_id)
  objective_count = _detect_objective_count(run_dir)
  html_text = _render_html(title, run_dir, assets, job_id, objective_count, file_base)
  return html_text, len(assets)


def _build_for_run(run_dir: Path, title: str) -> int:
  html_text, count = build_compact_dashboard_html(run_dir, title, job_id=None)
  out_path = run_dir / "dashboard_compact.html"
  out_path.write_text(html_text, encoding="utf-8")
  return count


def main() -> None:
  parser = argparse.ArgumentParser(description="Generate compact dashboards for RAVEN runs.")
  parser.add_argument("--run-dir", type=Path, help="Single run directory to render.")
  parser.add_argument(
    "--runs-root",
    type=Path,
    default=_default_runs_root(),
    help="Root folder containing run directories (default: repo webui_runs).",
  )
  parser.add_argument("--title", default="Output Dashboard", help="Dashboard title.")
  args = parser.parse_args()

  run_dir = args.run_dir.resolve() if args.run_dir else None
  runs_root = args.runs_root.resolve()
  if run_dir and not run_dir.exists():
    raise SystemExit(f"Run directory not found: {run_dir}")
  if not run_dir and not runs_root.exists():
    raise SystemExit(f"Runs root not found: {runs_root}")

  total = 0
  for target in _iter_run_dirs(run_dir, runs_root):
    try:
      count = _build_for_run(target, args.title)
      print(f"Wrote {target / 'dashboard_compact.html'} ({count} plots)")
      total += 1
    except Exception as exc:
      print(f"Skipped {target}: {exc}")
  print(f"Processed {total} run(s).")


if __name__ == "__main__":
  main()
