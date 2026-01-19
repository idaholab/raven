"""
Web UI scaffolding for PRLO reload optimization.

This package exposes a FastAPI application and lightweight
parsers that surface PRLO template information over a local
HTTP interface.  Use ``python -m webui.app`` to
launch the development server.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["DEFAULT_EXAMPLE_PROJECT"]

# Default example points to the AP1000 multi-objective study
# bundled with the repository.  Consumers can override this by
# passing explicit ``--project`` arguments when launching the
# server.
DEFAULT_EXAMPLE_PROJECT = Path(
  "plugins/PRLO/examples/AP1000/opt_multiobjective"
).resolve()
