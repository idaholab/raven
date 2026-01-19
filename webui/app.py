"""
Entry point for the PRLO web UI FastAPI application.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence
from urllib.parse import quote
import re

try:
  from fastapi import FastAPI, HTTPException
  from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
  from fastapi.middleware.cors import CORSMiddleware
  from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError as exc:  # pragma: no cover
  raise RuntimeError(
    "FastAPI is required to run the PRLO web UI backend. "
    "Install it in the active environment via 'pip install fastapi uvicorn'."
  ) from exc

import uvicorn

from .data import ProjectRepository
from .schema import (
  Health,
  InventoryResponse,
  KpiSeries,
  LayoutResponse,
  LayoutUpdate,
  ProjectDetail,
  ProjectSummary,
)
from .raven_entities import entity_options
from .run_manager import RunManager
from .dashboard_builder import DashboardItem
from .build_compact_dashboards import build_compact_dashboard_html
from .xml_builder import build_catalog, load_example_xml

LOGGER = logging.getLogger("prlo.webui")
_PROJECTS_ENV = "PRLO_WEBUI_PROJECTS"
_RUN_MANAGER: Optional[RunManager] = None
_PRLO_DASHBOARD_DIR = Path("plugins/PRLO/examples/AP1000_nthcycle/opt_multiobjective_50iter_50pop_nthcycle")
_PRLO_DASHBOARD_ENTRY = "dashboard_compact.html"


def create_app(repository: ProjectRepository) -> FastAPI:
  """
  Construct a FastAPI application bound to the given repository.
  """
  app = FastAPI(
    title="PRLO Reload Web UI",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
  )

  app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
  )

  @app.get("/api/health", response_model=Health)
  def health() -> Health:
    return Health()

  @app.get("/api/projects", response_model=List[ProjectSummary])
  def list_projects() -> List[ProjectSummary]:
    return repository.summaries()

  @app.get("/api/projects/{project_id}", response_model=ProjectDetail)
  def project_detail(project_id: str) -> ProjectDetail:
    try:
      return repository.detail(project_id)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found") from exc

  @app.get("/api/projects/{project_id}/layout", response_model=LayoutResponse)
  def project_layout(project_id: str) -> LayoutResponse:
    try:
      return repository.layout(project_id)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found") from exc

  @app.post("/api/projects/{project_id}/layout", response_model=LayoutResponse)
  def update_layout(project_id: str, payload: LayoutUpdate) -> LayoutResponse:
    # Persistence and optimizer integration will be handled in later iterations.
    LOGGER.info("Received layout update for %s (%d placements)", project_id, len(payload.placements))
    try:
      # Return the current layout until write-back is implemented.
      return repository.layout(project_id)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found") from exc

  @app.get("/api/projects/{project_id}/inventory", response_model=InventoryResponse)
  def project_inventory(project_id: str) -> InventoryResponse:
    try:
      return repository.inventory(project_id)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found") from exc

  @app.get("/api/projects/{project_id}/kpis", response_model=KpiSeries)
  def project_kpis(project_id: str) -> KpiSeries:
    try:
      return repository.kpis(project_id)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found") from exc

  @app.get("/api/xml-builder/catalog")
  def xml_builder_catalog() -> dict:
    return build_catalog()

  @app.get("/api/xml-builder/example", response_class=PlainTextResponse)
  def xml_builder_example(path: str) -> str:
    try:
      return load_example_xml(path)
    except ValueError as exc:
      raise HTTPException(status_code=400, detail=str(exc)) from exc

  @app.get("/api/xml-builder/entity-options")
  def xml_builder_entity_options(entity: str) -> dict:
    return {"entity": entity, "options": entity_options(entity)}

  @app.get("/api/xml-builder/output-dashboard")
  def xml_builder_output_dashboard() -> RedirectResponse:
    return RedirectResponse(url=f"/api/xml-builder/prlo-dashboard/{_PRLO_DASHBOARD_ENTRY}")

  @app.get("/api/xml-builder/prlo-dashboard/{path:path}")
  def xml_builder_prlo_dashboard_file(path: str) -> FileResponse:
    root = (_project_root() / _PRLO_DASHBOARD_DIR).resolve()
    requested = Path(path)
    if requested.is_absolute():
      raise HTTPException(status_code=400, detail="Absolute paths are not allowed.")
    candidate = (root / requested).resolve()
    try:
      candidate.relative_to(root)
    except ValueError:
      raise HTTPException(status_code=400, detail="Requested path is outside dashboard directory.") from None
    if not candidate.exists():
      raise HTTPException(status_code=404, detail="File not found.")
    if candidate.is_dir():
      raise HTTPException(status_code=400, detail="Requested path is a directory.")
    return FileResponse(candidate)

  @app.post("/api/xml-builder/run")
  def xml_builder_run(payload: dict) -> dict:
    xml_text = payload.get("xml") if isinstance(payload, dict) else None
    if not isinstance(xml_text, str) or not xml_text.strip():
      raise HTTPException(status_code=400, detail="Missing 'xml' payload.")
    context = payload.get("context_path") if isinstance(payload, dict) else None
    context_path = None
    if isinstance(context, str) and context.strip():
      candidate = (_project_root() / context.strip()).resolve()
      # Must stay inside the repo.
      try:
        candidate.relative_to(_project_root().resolve())
      except ValueError:
        raise HTTPException(status_code=400, detail="context_path must be within the repository.") from None
      context_path = candidate
    global _RUN_MANAGER
    if _RUN_MANAGER is None:
      _RUN_MANAGER = RunManager(repo_root=_project_root(), conda_env="raven_libraries")
    job = _RUN_MANAGER.submit(xml_text, context_path=context_path)
    return {"job_id": job.job_id, "status": job.status, "job_dir": job.workdir, "raven_workdir": job.raven_workdir}

  @app.get("/api/xml-builder/run/{job_id}")
  def xml_builder_run_status(job_id: str, tail_lines: int = 300) -> dict:
    global _RUN_MANAGER
    if _RUN_MANAGER is None:
      raise HTTPException(status_code=404, detail="No runs started.")
    job = _RUN_MANAGER.get(job_id)
    if job is None:
      raise HTTPException(status_code=404, detail=f"Run '{job_id}' not found.")
    tail = _RUN_MANAGER.tail_log(job_id, max_lines=max(50, min(int(tail_lines), 2000)))
    dashboard_files: List[str] = []
    dashboard_candidates: List[dict] = []
    if job.raven_workdir:
      try:
        base = Path(job.raven_workdir)
        if base.exists():
          html_paths = sorted(base.rglob("*.html"))
          dashboard_files = [str(path.relative_to(base)) for path in html_paths[:50]]

          def score_path(rel: str, size_bytes: int) -> float:
            name = rel.lower()
            score = 0.0
            if "dashboard_compact" in name:
              score += 120
            if "dashboard" in name:
              score += 60
            if name.endswith("index.html") or "/index.html" in name:
              score += 45
            if "report" in name or "summary" in name:
              score += 25
            if "overview" in name:
              score += 20
            # Prefer larger HTML as "multi-plot" candidates.
            score += min(40.0, max(0.0, (size_bytes / 1024.0) / 25.0))
            return score

          for path in html_paths:
            try:
              rel = str(path.relative_to(base))
              size_bytes = path.stat().st_size
              title = ""
              try:
                head = path.read_text(encoding="utf-8", errors="ignore")[:8192]
                match = re.search(r"<title>(.*?)</title>", head, flags=re.IGNORECASE | re.DOTALL)
                if match:
                  title = " ".join(match.group(1).split())[:120]
              except Exception:
                title = ""
              dashboard_candidates.append(
                {
                  "path": rel,
                  "title": title,
                  "size_bytes": size_bytes,
                  "score": score_path(rel, size_bytes),
                }
              )
            except Exception:
              continue
          dashboard_candidates.sort(key=lambda item: item.get("score", 0.0), reverse=True)
          dashboard_candidates = dashboard_candidates[:12]
      except Exception:
        dashboard_files = []
        dashboard_candidates = []
    return {
      "job_id": job.job_id,
      "status": job.status,
      "returncode": job.returncode,
      "created_at": job.created_at,
      "started_at": job.started_at,
      "finished_at": job.finished_at,
      "job_dir": job.workdir,
      "context_dir": job.context_dir,
      "raven_workdir": job.raven_workdir,
      "input_path": job.input_path,
      "log_path": job.log_path,
      "error": job.error,
      "tail": tail,
      "dashboard_files": dashboard_files,
      "dashboard_candidates": dashboard_candidates,
      "dashboard_url": f"/api/xml-builder/run/{job.job_id}/dashboard",
    }

  @app.get("/api/xml-builder/run/{job_id}/outputs")
  def xml_builder_run_outputs(job_id: str, limit: int = 200) -> dict:
    global _RUN_MANAGER
    if _RUN_MANAGER is None:
      raise HTTPException(status_code=404, detail="No runs started.")
    job = _RUN_MANAGER.get(job_id)
    if job is None:
      raise HTTPException(status_code=404, detail=f"Run '{job_id}' not found.")
    roots: List[Path] = []
    if job.raven_workdir:
      roots.append(Path(job.raven_workdir))
    roots.append(Path(job.workdir))

    max_items = max(10, min(int(limit), 1000))
    items: List[dict] = []
    seen = set()

    def is_dashboard(path: Path) -> bool:
      name = path.name.lower()
      return name in {"dashboard.html", "dashboard_compact.html"}

    def classify_kind(path: Path) -> str:
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

    def safe_size(path: Path) -> int:
      try:
        return path.stat().st_size
      except OSError:
        return 0

    for root in roots:
      try:
        base = root.resolve()
      except Exception:
        continue
      if not base.exists():
        continue
      for path in sorted(base.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
          continue
        if is_dashboard(path):
          continue
        try:
          rel = str(path.relative_to(base))
        except Exception:
          continue
        key = (str(base), rel)
        if key in seen:
          continue
        seen.add(key)
        items.append(
          {
            "path": rel,
            "label": path.name,
            "kind": classify_kind(path),
            "size": safe_size(path),
          }
        )
        if len(items) >= max_items:
          break
      if len(items) >= max_items:
        break

    return {"job_id": job.job_id, "items": items}

  @app.get("/api/xml-builder/run/{job_id}/dashboard")
  def xml_builder_run_dashboard(job_id: str) -> FileResponse:
    global _RUN_MANAGER
    if _RUN_MANAGER is None:
      raise HTTPException(status_code=404, detail="No runs started.")
    job = _RUN_MANAGER.get(job_id)
    if job is None:
      raise HTTPException(status_code=404, detail=f"Run '{job_id}' not found.")
    roots: List[Path] = []
    if job.raven_workdir:
      roots.append(Path(job.raven_workdir))
    roots.append(Path(job.workdir))

    def is_dashboard(path: Path) -> bool:
      name = path.name.lower()
      return name in {"dashboard.html", "dashboard_compact.html"}

    def classify_kind(path: Path) -> str:
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

    def safe_size(path: Path) -> int:
      try:
        return path.stat().st_size
      except OSError:
        return 0

    run_dir = Path(job.raven_workdir) if job.raven_workdir else Path(job.workdir)
    try:
      run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
      raise HTTPException(status_code=500, detail=str(exc)) from exc

    html, _count = build_compact_dashboard_html(run_dir, title="Output Dashboard", job_id=job.job_id)
    out_path = run_dir / "dashboard_compact.html"
    try:
      out_path.write_text(html, encoding="utf-8")
    except OSError as exc:
      raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileResponse(out_path)

  @app.get("/api/xml-builder/run/{job_id}/file")
  def xml_builder_run_file(job_id: str, path: str) -> FileResponse:
    global _RUN_MANAGER
    if _RUN_MANAGER is None:
      raise HTTPException(status_code=404, detail="No runs started.")
    job = _RUN_MANAGER.get(job_id)
    if job is None:
      raise HTTPException(status_code=404, detail=f"Run '{job_id}' not found.")
    if not isinstance(path, str) or not path.strip():
      raise HTTPException(status_code=400, detail="Missing 'path' query parameter.")
    # Only allow reads inside the discovered RAVEN working directory (preferred) or run folder.
    roots: List[Path] = []
    if job.raven_workdir:
      roots.append(Path(job.raven_workdir))
    roots.append(Path(job.workdir))
    requested = Path(path)
    if requested.is_absolute():
      raise HTTPException(status_code=400, detail="Absolute paths are not allowed.")
    for root in roots:
      try:
        candidate = (root / requested).resolve()
        candidate.relative_to(root.resolve())
      except Exception:
        continue
      if not candidate.exists():
        raise HTTPException(status_code=404, detail="File not found.")
      if candidate.is_dir():
        raise HTTPException(status_code=400, detail="Requested path is a directory.")
      return FileResponse(candidate)
    raise HTTPException(status_code=400, detail="Requested path is outside allowed run directories.")

  def _runs_root() -> Path:
    return _project_root() / "webui_runs"

  def _resolve_run_dir(run_name: str) -> Path:
    if not run_name or "/" in run_name or "\\" in run_name:
      raise HTTPException(status_code=400, detail="Invalid run name.")
    root = _runs_root().resolve()
    candidate = (root / run_name).resolve()
    try:
      candidate.relative_to(root)
    except Exception:
      raise HTTPException(status_code=400, detail="Run path is outside runs root.")
    if not candidate.exists() or not candidate.is_dir():
      raise HTTPException(status_code=404, detail="Run directory not found.")
    return candidate

  def _read_run_display_name(run_dir: Path) -> Optional[str]:
    input_xml = run_dir / "input.xml"
    candidates = [input_xml] if input_xml.exists() else []
    candidates.extend(sorted(run_dir.glob(".webui_input_*.xml")))
    for candidate in candidates:
      try:
        text = candidate.read_text(encoding="utf-8", errors="ignore")
      except OSError:
        continue
      match = re.search(r"<WorkingDir>(.*?)</WorkingDir>", text, flags=re.IGNORECASE | re.DOTALL)
      if match:
        value = match.group(1).strip()
        if value:
          return Path(value).name
    return None

  @app.get("/api/xml-builder/runs")
  def xml_builder_runs(limit: int = 200) -> dict:
    root = _runs_root()
    if not root.exists():
      return {"runs": []}
    runs = []
    max_items = max(1, min(int(limit), 500))
    for child in sorted(root.iterdir(), key=lambda p: p.name):
      if not child.is_dir() or child.name.startswith("_"):
        continue
      display_name = _read_run_display_name(child) or child.name
      runs.append(
        {
          "name": child.name,
          "display_name": display_name,
          "path": str(child),
        }
      )
      if len(runs) >= max_items:
        break
    return {"runs": runs}

  @app.get("/api/xml-builder/run-folder/{run_name}/dashboard")
  def xml_builder_run_folder_dashboard(run_name: str, job_id: Optional[str] = None) -> FileResponse:
    run_dir = _resolve_run_dir(run_name)
    file_base = f"/api/xml-builder/run-folder/{run_name}/file"
    html, _count = build_compact_dashboard_html(
      run_dir,
      title="Output Dashboard",
      job_id=job_id,
      file_base=file_base,
    )
    out_path = run_dir / "dashboard_compact.html"
    try:
      out_path.write_text(html, encoding="utf-8")
    except OSError as exc:
      raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileResponse(out_path)

  @app.get("/api/xml-builder/run-folder/{run_name}/file")
  def xml_builder_run_folder_file(run_name: str, path: str) -> FileResponse:
    run_dir = _resolve_run_dir(run_name)
    if not isinstance(path, str) or not path.strip():
      raise HTTPException(status_code=400, detail="Missing 'path' query parameter.")
    requested = Path(path)
    if requested.is_absolute():
      raise HTTPException(status_code=400, detail="Absolute paths are not allowed.")
    try:
      candidate = (run_dir / requested).resolve()
      candidate.relative_to(run_dir.resolve())
    except Exception:
      raise HTTPException(status_code=400, detail="Requested path is outside allowed run directory.")
    if not candidate.exists():
      raise HTTPException(status_code=404, detail="File not found.")
    if candidate.is_dir():
      raise HTTPException(status_code=400, detail="Requested path is a directory.")
    return FileResponse(candidate)

  def _resolve_any_run_dir(path: str) -> Path:
    if not path or not isinstance(path, str):
      raise HTTPException(status_code=400, detail="Missing 'path' query parameter.")
    root = Path(path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
      raise HTTPException(status_code=404, detail="Run directory not found.")
    return root

  @app.get("/api/xml-builder/run-path/dashboard")
  def xml_builder_run_path_dashboard(path: str) -> FileResponse:
    run_dir = _resolve_any_run_dir(path)
    file_base = f"/api/xml-builder/run-path/file?root={quote(path, safe='')}"
    html, _count = build_compact_dashboard_html(
      run_dir,
      title="Output Dashboard",
      job_id=None,
      file_base=file_base,
    )
    out_path = run_dir / "dashboard_compact.html"
    try:
      out_path.write_text(html, encoding="utf-8")
    except OSError as exc:
      raise HTTPException(status_code=500, detail=str(exc)) from exc
    return FileResponse(out_path)

  @app.get("/api/xml-builder/run-path/file")
  def xml_builder_run_path_file(root: str, path: str) -> FileResponse:
    run_dir = _resolve_any_run_dir(root)
    if not isinstance(path, str) or not path.strip():
      raise HTTPException(status_code=400, detail="Missing 'path' query parameter.")
    requested = Path(path)
    if requested.is_absolute():
      raise HTTPException(status_code=400, detail="Absolute paths are not allowed.")
    try:
      candidate = (run_dir / requested).resolve()
      candidate.relative_to(run_dir.resolve())
    except Exception:
      raise HTTPException(status_code=400, detail="Requested path is outside allowed run directory.")
    if not candidate.exists():
      raise HTTPException(status_code=404, detail="File not found.")
    if candidate.is_dir():
      raise HTTPException(status_code=400, detail="Requested path is a directory.")
    return FileResponse(candidate)

  @app.get("/api/xml-builder/list-dirs")
  def xml_builder_list_dirs(path: Optional[str] = None) -> dict:
    root = Path(path).expanduser().resolve() if path else Path.cwd().resolve()
    if not root.exists() or not root.is_dir():
      raise HTTPException(status_code=404, detail="Directory not found.")
    dirs = []
    try:
      for child in root.iterdir():
        if child.is_dir():
          dirs.append({"name": child.name, "path": str(child)})
    except OSError as exc:
      raise HTTPException(status_code=500, detail=str(exc)) from exc
    dirs.sort(key=lambda item: item["name"].lower())
    parent = str(root.parent) if root.parent != root else None
    return {"path": str(root), "parent": parent, "dirs": dirs}

  static_dir = Path(__file__).with_name("static")
  if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    builder_index = static_dir / "xml_builder" / "index.html"
    if builder_index.exists():
      @app.get("/", include_in_schema=False)
      def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/xml-builder")

      @app.get("/xml-builder", include_in_schema=False)
      def xml_builder_ui() -> FileResponse:
        return FileResponse(builder_index)

      @app.get("/xml-builder/", include_in_schema=False)
      def xml_builder_ui_trailing_slash() -> FileResponse:
        return FileResponse(builder_index)

  return app


def create_app_from_env() -> FastAPI:
  """
  Uvicorn factory entrypoint for reload mode.

  Uvicorn requires an import string when using ``--reload``. We use this factory
  and pass configuration via environment variables.
  """
  project_paths: Optional[List[Path]] = None
  raw = os.environ.get(_PROJECTS_ENV, "").strip()
  if raw:
    project_paths = [Path(item) for item in raw.split(os.pathsep) if item]
  repository = ProjectRepository(project_paths=project_paths)
  return create_app(repository)


def _project_root() -> Path:
  current = Path(__file__).resolve()
  for parent in [current.parent] + list(current.parents):
    if (parent / "ravenframework").is_dir():
      return parent
  return current.parents[1]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Launch the PRLO reload web UI backend.")
  parser.add_argument(
    "--project",
    action="append",
    dest="projects",
    type=Path,
    help="Project directory containing sim3-param*.xml (may be specified multiple times).",
  )
  parser.add_argument("--host", default="127.0.0.1", help="Bind address for the server.")
  parser.add_argument("--port", type=int, default=8750, help="TCP port for the server.")
  parser.add_argument(
    "--reload",
    action="store_true",
    help="Enable autoreload (development only).",
  )
  parser.add_argument(
    "--list",
    action="store_true",
    help="List discovered projects and exit.",
  )
  return parser.parse_args(argv)


def _configure_logging() -> None:
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
  )


def main(argv: Optional[Sequence[str]] = None) -> None:
  _configure_logging()
  args = _parse_args(argv)
  if args.reload:
    if args.projects:
      os.environ[_PROJECTS_ENV] = os.pathsep.join(str(path) for path in args.projects)
    else:
      os.environ.pop(_PROJECTS_ENV, None)
    uvicorn.run(
      "webui.app:create_app_from_env",
      factory=True,
      host=args.host,
      port=args.port,
      reload=True,
      log_level="info",
    )
    return

  repository = ProjectRepository(project_paths=args.projects)
  if args.list:
    summaries = repository.summaries()
    if not summaries:
      print("No projects discovered.", file=sys.stderr)
      return
    for summary in summaries:
      print(f"{summary.id}\t{summary.name}\t{summary.path}")
    return

  app = create_app(repository)
  uvicorn.run(
    app,
    host=args.host,
    port=args.port,
    reload=False,
    log_level="info",
  )


if __name__ == "__main__":
  main()
