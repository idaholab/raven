"""
Simple job runner for executing RAVEN runs from the web UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
from threading import Lock, Thread
from typing import Dict, Optional
import subprocess
import shutil
import uuid


@dataclass
class RunJob:
  job_id: str
  status: str  # queued|running|done|error
  created_at: str
  started_at: Optional[str]
  finished_at: Optional[str]
  returncode: Optional[int]
  workdir: str
  context_dir: Optional[str]
  raven_workdir: Optional[str]
  input_path: str
  log_path: str
  command: str
  error: Optional[str]


class RunManager:
  def __init__(self, repo_root: Path, conda_env: str = "raven_libraries") -> None:
    self._repo_root = repo_root
    self._conda_env = conda_env
    self._lock = Lock()
    self._jobs: Dict[str, RunJob] = {}
    self._conda_bin = self._resolve_conda_bin()
    self._runs_root = self._repo_root / "webui_runs"
    self._jobs_root = self._runs_root / "_jobs"

  def _resolve_conda_bin(self) -> Optional[Path]:
    """
    Resolve a conda executable path without relying on shell init scripts.
    """
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
      candidate = Path(conda_exe)
      if candidate.exists() and os.access(str(candidate), os.X_OK):
        return candidate

    which = shutil.which("conda")
    if which:
      candidate = Path(which)
      if candidate.exists() and os.access(str(candidate), os.X_OK):
        return candidate

    home = Path.home()
    for guess in [
      home / "miniconda3" / "bin" / "conda",
      home / "miniforge3" / "bin" / "conda",
      home / "mambaforge" / "bin" / "conda",
    ]:
      if guess.exists() and os.access(str(guess), os.X_OK):
        return guess
    return None

  def submit(self, xml_text: str, context_path: Optional[Path] = None) -> RunJob:
    job_id = uuid.uuid4().hex
    short_id = job_id[:8]
    context_dir: Optional[Path] = None
    if context_path is not None:
      context_path = context_path.resolve()
      try:
        context_path.relative_to(self._repo_root.resolve())
      except ValueError:
        context_path = None
      else:
        if context_path.is_dir():
          context_dir = context_path
        elif context_path.is_file():
          context_dir = context_path.parent
    # Store job metadata/logs under a stable jobs folder.
    run_root = self._jobs_root / job_id
    run_root.mkdir(parents=True, exist_ok=True)
    # Place the input file in the *context directory* so relative paths inside the XML resolve
    # the same way they do when running the original input deck from its folder.
    input_path = (context_dir / f".webui_input_{job_id}.xml") if context_dir else (run_root / "input.xml")
    log_path = run_root / "run.log"
    raven_workdir: Optional[Path] = None
    patched_xml = xml_text
    # Always put the RAVEN WorkingDir under repo_root/webui_runs for easy discovery.
    match = re.search(
      r"(<RunInfo[^>]*>.*?<WorkingDir>)(.*?)(</WorkingDir>)",
      patched_xml,
      flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
      original = match.group(2).strip()
      base = original or "webui_run"
    else:
      base = "webui_run"
    candidate = base
    target = (self._runs_root / candidate).resolve()
    if target.exists():
      candidate = f"{base}_webui_{short_id}"
      target = (self._runs_root / candidate).resolve()
    raven_workdir = target
    if match:
      # Use an absolute path to avoid dependency on the current working directory.
      patched_xml = patched_xml[: match.start(2)] + str(target) + patched_xml[match.end(2) :]
    input_path.write_text(patched_xml, encoding="utf-8")

    rel_input = input_path.relative_to(self._repo_root)
    if self._conda_bin is None:
      command = "conda (not found)"
    else:
      command = (
        f"{self._conda_bin} run -n {self._conda_env} --no-capture-output "
        f"./raven_framework {rel_input}"
      )

    job = RunJob(
      job_id=job_id,
      status="queued",
      created_at=datetime.utcnow().isoformat() + "Z",
      started_at=None,
      finished_at=None,
      returncode=None,
      workdir=str(run_root),
      context_dir=str(context_dir) if context_dir else None,
      raven_workdir=str(raven_workdir) if raven_workdir else None,
      input_path=str(input_path),
      log_path=str(log_path),
      command=command,
      error=None,
    )

    with self._lock:
      self._jobs[job_id] = job

    thread = Thread(target=self._run_job, args=(job_id,), daemon=True)
    thread.start()
    return job

  def get(self, job_id: str) -> Optional[RunJob]:
    with self._lock:
      return self._jobs.get(job_id)

  def tail_log(self, job_id: str, max_lines: int = 300) -> str:
    job = self.get(job_id)
    if job is None:
      return ""
    log_path = Path(job.log_path)
    if not log_path.exists():
      return ""
    try:
      lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
      return ""
    return "\n".join(lines[-max_lines:]) + ("\n" if lines else "")

  def _run_job(self, job_id: str) -> None:
    job = self.get(job_id)
    if job is None:
      return

    with self._lock:
      job.status = "running"
      job.started_at = datetime.utcnow().isoformat() + "Z"

    log_path = Path(job.log_path)
    try:
      with log_path.open("w", encoding="utf-8") as log_file:
        if self._conda_bin is None:
          raise RuntimeError(
            "Could not locate a conda executable. "
            "Start the web UI from a terminal where `conda` is available, or set CONDA_EXE."
          )
        args = [
          str(self._conda_bin),
          "run",
          "-n",
          self._conda_env,
          "--no-capture-output",
          "./raven_framework",
          str(Path(job.input_path).relative_to(self._repo_root)),
        ]
        process = subprocess.Popen(args, cwd=str(self._repo_root), stdout=log_file, stderr=subprocess.STDOUT)
        rc = process.wait()
    except Exception as exc:
      with self._lock:
        job.status = "error"
        job.error = str(exc)
        job.returncode = -1
        job.finished_at = datetime.utcnow().isoformat() + "Z"
      return

    with self._lock:
      job.returncode = rc
      job.finished_at = datetime.utcnow().isoformat() + "Z"
      job.status = "done" if rc == 0 else "error"
