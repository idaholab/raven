#!/usr/bin/env python3
"""
Sync InputData specs, generated XSDs, and the audit baseline JSON.

This script regenerates:
- per-entity XSDs under developer_tools/XSDSchemas/generated
- full Simulation XSD (default: developer_tools/XSDSchemas/generated/raven.xsd)
- audit baseline JSON (developer_tools/audit_input_specs.json)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd, cwd=None, strict=True):
  print("Running:", " ".join(cmd))
  try:
    subprocess.check_call(cmd, cwd=cwd)
    return True
  except subprocess.CalledProcessError as exc:
    if strict:
      raise
    print("WARNING: command failed (continuing):", exc)
    return False


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--out-json",
      default=str(REPO_ROOT / "developer_tools" / "audit_input_specs.json"),
      help="Path to write audit baseline JSON.",
  )
  parser.add_argument(
      "--xsd-dir",
      default=str(REPO_ROOT / "developer_tools" / "XSDSchemas" / "generated"),
      help="Directory to write per-entity generated XSDs.",
  )
  parser.add_argument(
      "--full-xsd",
      default=str(REPO_ROOT / "developer_tools" / "XSDSchemas" / "generated" / "raven.xsd"),
      help="Path to write the full Simulation XSD.",
  )
  parser.add_argument(
      "--skip-xsd",
      action="store_true",
      help="Skip regenerating XSDs (only update audit JSON).",
  )
  parser.add_argument(
      "--strict",
      action="store_true",
      help="Fail if any sync step fails.",
  )
  args = parser.parse_args()

  python = sys.executable
  if not args.skip_xsd:
    gen_entity = REPO_ROOT / "developer_tools" / "generate_xsd_from_inputdata.py"
    _run([python, str(gen_entity), "--out", args.xsd_dir], cwd=str(REPO_ROOT), strict=args.strict)
    gen_full = REPO_ROOT / "developer_tools" / "gen_xsd.py"
    _run([python, str(gen_full), args.full_xsd], cwd=str(REPO_ROOT), strict=args.strict)

  audit = REPO_ROOT / "developer_tools" / "audit_input_specs.py"
  _run([python, str(audit), "--json", args.out_json, "--no-print"], cwd=str(REPO_ROOT), strict=True)

  print("Sync complete.")


if __name__ == "__main__":
  main()
