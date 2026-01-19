"""
Helpers for turning PRLO example RAVEN XML inputs into reusable UI snippets.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from xml.etree import ElementTree as ET


def _examples_root() -> Path:
  for candidate in ("plugins/PRLO/examples", "plugins/prlo/examples"):
    path = Path(candidate).resolve()
    if path.exists():
      return path
  return Path("plugins/PRLO/examples").resolve()


def _repo_root() -> Path:
  examples_root = _examples_root()
  # <repo>/plugins/PRLO/examples
  return examples_root.parents[2]


def _is_raven_simulation_input(xml_path: Path) -> bool:
  """
  Detect RAVEN XML inputs by scanning the header for the ``<Simulation`` tag.
  """
  try:
    with xml_path.open("rb") as handle:
      head = handle.read(4096)
  except OSError:
    return False
  return b"<Simulation" in head


def discover_example_inputs() -> List[Path]:
  """
  Locate PRLO example XML inputs that are RAVEN Simulation decks.
  """
  root = _examples_root()
  inputs: List[Path] = []
  for xml_path in root.rglob("*.xml"):
    if _is_raven_simulation_input(xml_path):
      inputs.append(xml_path)
  inputs.sort()
  return inputs


def _parser() -> ET.XMLParser:
  return ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))


def _pretty_xml(element: ET.Element) -> str:
  raw = ET.tostring(element, encoding="unicode")
  try:
    from xml.dom import minidom
  except ImportError:
    return raw
  try:
    doc = minidom.parseString(raw)
  except Exception:
    return raw
  pretty = doc.toprettyxml(indent="  ")
  lines = [line for line in pretty.splitlines() if line.strip()]
  if lines and lines[0].startswith("<?xml"):
    lines = lines[1:]
  return "\n".join(lines)


def _snippet_id(*parts: str) -> str:
  digest = hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()
  return digest[:16]


def _snippet_label(node: ET.Element) -> str:
  name = node.attrib.get("name") or node.attrib.get("type") or node.attrib.get("class")
  if name:
    return name
  return ""


def _iter_section_snippets(
  root: ET.Element,
  source_path: Path,
) -> Iterable[Dict[str, object]]:
  for section in list(root):
    if not isinstance(section.tag, str):
      continue
    section_tag = section.tag
    for block in list(section):
      if not isinstance(block.tag, str):
        continue
      block_tag = block.tag
      label = _snippet_label(block)
      snippet_xml = _pretty_xml(block)
      snippet = {
        "id": _snippet_id(str(source_path), section_tag, block_tag, label, snippet_xml),
        "section": section_tag,
        "tag": block_tag,
        "label": label,
        "name": block.attrib.get("name"),
        "source": source_path.as_posix(),
        "xml": snippet_xml,
      }
      yield snippet


@lru_cache(maxsize=1)
def build_catalog() -> Dict[str, object]:
  """
  Build an API payload describing available example inputs and XML snippets.

  Cached for the process lifetime.
  """
  example_inputs = discover_example_inputs()
  repo_root = _repo_root()
  snippets: List[Dict[str, object]] = []
  for xml_path in example_inputs:
    try:
      tree = ET.parse(xml_path, parser=_parser())
    except ET.ParseError:
      continue
    root = tree.getroot()
    if root.tag != "Simulation":
      continue
    source_path = xml_path.relative_to(repo_root)
    snippets.extend(list(_iter_section_snippets(root, source_path)))

  generated_at = datetime.now(timezone.utc).isoformat()
  return {
    "generated_at": generated_at,
    "examples": [
      {
        "path": path.relative_to(repo_root).as_posix(),
        "name": path.stem,
      }
      for path in example_inputs
    ],
    "snippets": snippets,
  }


def load_example_xml(example_path: str) -> str:
  root = _examples_root()
  candidate = Path(example_path).expanduser()
  if not candidate.is_absolute():
    candidate = (_repo_root() / candidate)
  target = candidate.resolve()
  if root not in target.parents and target != root:
    raise ValueError("Example path must live under plugins/PRLO/examples.")
  if target.suffix.lower() != ".xml":
    raise ValueError("Example path must be an XML file.")
  if not _is_raven_simulation_input(target):
    raise ValueError("Example path does not appear to be a RAVEN Simulation input.")
  return target.read_text(encoding="utf-8")


def catalog_json() -> str:
  return json.dumps(build_catalog(), indent=2, sort_keys=True)
