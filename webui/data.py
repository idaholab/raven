"""
Utilities for loading PRLO project artifacts into API-friendly structures.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
from xml.etree import ElementTree as ET

from . import DEFAULT_EXAMPLE_PROJECT
from .schema import (
  AssemblyType,
  InventoryItem,
  InventoryResponse,
  KpiSample,
  KpiSeries,
  LayoutPlacement,
  LayoutResponse,
  ProjectDetail,
  ProjectSummary,
  SlotDefinition,
)


def _strip_quotes(value: str) -> str:
  """
  Remove surrounding quotes and whitespace from a string.
  """
  return value.strip().strip("'\"")


def _slugify(text: str) -> str:
  """
  Generate a filesystem-safe slug from a piece of text.
  """
  lower = text.lower()
  slug = re.sub(r"[^a-z0-9]+", "-", lower).strip("-")
  return slug or "project"


@dataclass
class Sim3Template:
  """
  Representation of a SIMULATE3 template definition.
  """

  prefix: str
  slots: List[SlotDefinition]
  assembly_types: List[AssemblyType]

  @classmethod
  def from_file(cls, xml_path: Path) -> "Sim3Template":
    if not xml_path.exists():
      raise FileNotFoundError(f"Template file not found: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    prefix_node = root.find("Prefix_indicator")
    prefix = ""
    if prefix_node is not None and prefix_node.text:
      prefix = _strip_quotes(prefix_node.text)

    slot_nodes = root.findall(".//Var-list/Var")
    slots: List[SlotDefinition] = []
    for node in slot_nodes:
      raven_id = int(node.attrib.get("ravenid", "0").strip())
      value = node.attrib.get("value", "")
      positions = value.split()
      template_label = f"{prefix}loc{raven_id + 1}$"
      slots.append(
        SlotDefinition(
          slot_id=raven_id,
          variable=f"loc{raven_id + 1}",
          template_label=template_label,
          symmetry_positions=positions,
        )
      )
    slots.sort(key=lambda slot: slot.slot_id)

    assembly_types: List[AssemblyType] = []
    for idx, node in enumerate(root.findall(".//Fresh_FA_list/FreshFA"), start=1):
      attributes = {k: _strip_quotes(v) for k, v in node.attrib.items()}
      assembly_types.append(
        AssemblyType(
          id=idx,
          name=attributes.get("name", f"FreshFA-{idx}"),
          serial_label=attributes.get("serial_label"),
          assembly_type=attributes.get("type"),
          attributes=attributes,
        )
      )

    return cls(prefix=prefix, slots=slots, assembly_types=assembly_types)


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
  """
  Read a CSV file and return a list of dictionaries preserving order.
  """
  with csv_path.open("r", newline="") as handle:
    reader = csv.DictReader(handle)
    return [row for row in reader]


def _select_layout_row(rows: Iterable[Dict[str, str]]) -> Optional[Dict[str, str]]:
  """
  Choose the most relevant row for layout reconstruction.
  """
  final_rows = [
    row for row in rows if row.get("accepted", "").strip().lower() == "final"
  ]
  if final_rows:
    return final_rows[-1]
  rows = list(rows)
  if rows:
    return rows[-1]
  return None


def _parse_float(value: Optional[str]) -> Optional[float]:
  if value is None or value == "":
    return None
  try:
    return float(value)
  except ValueError:
    return None


def _parse_int(value: Optional[str]) -> Optional[int]:
  number = _parse_float(value)
  if number is None:
    return None
  return int(number)


def _derive_symmetry(slots: Sequence[SlotDefinition]) -> Optional[str]:
  """
  Infer symmetry mode based on slot multiplicity.
  """
  counts = [len(slot.symmetry_positions) for slot in slots if slot.symmetry_positions]
  if not counts:
    return None
  most_common = max(set(counts), key=counts.count)
  if most_common == 1:
    return "full-core"
  if most_common == 4:
    return "quadrant"
  if most_common == 8:
    return "octant"
  return None


@dataclass
class ProjectRecord:
  """
  Wrapper around a project directory on disk.
  """

  id: str
  name: str
  path: Path
  template_path: Path
  sample_path: Optional[Path]
  opt_export_path: Optional[Path]
  template: Sim3Template

  @classmethod
  def from_path(cls, project_path: Path) -> "ProjectRecord":
    project_path = project_path.resolve()
    template_path = cls._find_one(project_path, "sim3-param*.xml")
    template = Sim3Template.from_file(template_path)
    sample_path = cls._find_one(project_path, "sample.inp", required=False)
    opt_export_path = cls._find_one(project_path, "opt_export_0.csv", required=False)
    name = project_path.name
    project_id = _slugify(name)
    return cls(
      id=project_id,
      name=name,
      path=project_path,
      template_path=template_path,
      sample_path=sample_path,
      opt_export_path=opt_export_path,
      template=template,
    )

  @staticmethod
  def _find_one(
    base_path: Path,
    pattern: str,
    required: bool = True,
  ) -> Optional[Path]:
    matches = list(base_path.glob(pattern))
    if matches:
      return matches[0]
    if required:
      raise FileNotFoundError(f"Could not find '{pattern}' under {base_path}")
    return None

  def summary(self) -> ProjectSummary:
    symmetry = _derive_symmetry(self.template.slots)
    core_type = "PWR" if "pwr" in self.name.lower() or "ap1000" in self.name.lower() else None
    return ProjectSummary(
      id=self.id,
      name=self.name,
      path=str(self.path),
      core_type=core_type,
      symmetry=symmetry,
    )

  def detail(self) -> ProjectDetail:
    summary = self.summary()
    return ProjectDetail(**summary.dict(), updated_at=datetime.utcfromtimestamp(self.path.stat().st_mtime))

  def layout(self) -> LayoutResponse:
    placements: List[LayoutPlacement] = []
    rows: List[Dict[str, str]] = []
    if self.opt_export_path and self.opt_export_path.exists():
      rows = _read_csv_rows(self.opt_export_path)
    layout_row = _select_layout_row(rows)
    if layout_row is None and rows:
      layout_row = rows[-1]
    for slot in self.template.slots:
      value = None
      if layout_row:
        value = layout_row.get(slot.variable)
      assembly_id = _parse_int(value) if value is not None else None
      placements.append(
        LayoutPlacement(
          slot_id=slot.slot_id,
          assembly_id=assembly_id,
          label=slot.variable,
          source=self.opt_export_path.name if self.opt_export_path else None,
        )
      )
    symmetry = _derive_symmetry(self.template.slots)
    return LayoutResponse(
      project_id=self.id,
      symmetry=symmetry,
      slots=self.template.slots,
      placements=placements,
    )

  def inventory(self) -> InventoryResponse:
    assembly_types = self.template.assembly_types
    items: List[InventoryItem] = []
    for idx, assembly in enumerate(assembly_types, start=1):
      quantity = int(assembly.attributes.get("quantity", "0"))
      items.append(
        InventoryItem(
          id=idx,
          assembly_type_id=assembly.id,
          quantity=quantity,
          status="fresh" if quantity else "empty",
        )
      )
    return InventoryResponse(
      project_id=self.id,
      assembly_types=assembly_types,
      items=items,
    )

  def kpis(self) -> KpiSeries:
    samples: List[KpiSample] = []
    if self.opt_export_path and self.opt_export_path.exists():
      for row in _read_csv_rows(self.opt_export_path):
        objective_values: Dict[str, float] = {}
        constraints: Dict[str, float] = {}
        for key in ("pin_peaking", "MaxFDH", "max_boron", "MaxEFPD"):
          value = _parse_float(row.get(key))
          if value is not None:
            objective_values[key] = value
        for key in ("FitnessEvaluation_MaxFDH", "FitnessEvaluation_pin_peaking"):
          value = _parse_float(row.get(key))
          if value is not None:
            constraints[key] = value
        metadata = {
          "working_dir": row.get("working_dir", ""),
          "accepted": row.get("accepted", ""),
        }
        samples.append(
          KpiSample(
            batch_id=_parse_int(row.get("batchId")),
            iteration=_parse_int(row.get("iteration")),
            rank=_parse_int(row.get("rank")),
            objective_values=objective_values,
            constraints=constraints,
            metadata=metadata,
          )
        )
    return KpiSeries(project_id=self.id, samples=samples)


class ProjectRepository:
  """
  Holds a collection of project records.
  """

  def __init__(self, project_paths: Optional[Sequence[Path]] = None):
    if not project_paths:
      project_paths = [DEFAULT_EXAMPLE_PROJECT]
    self._records: Dict[str, ProjectRecord] = {}
    for path in project_paths:
      try:
        record = ProjectRecord.from_path(path)
      except FileNotFoundError:
        continue
      slug = record.id
      # Ensure slug uniqueness.
      if slug in self._records:
        suffix = 1
        while f"{slug}-{suffix}" in self._records:
          suffix += 1
        record.id = f"{slug}-{suffix}"
      self._records[record.id] = record

  def summaries(self) -> List[ProjectSummary]:
    return [record.summary() for record in self._records.values()]

  def detail(self, project_id: str) -> ProjectDetail:
    record = self._records.get(project_id)
    if record is None:
      raise KeyError(project_id)
    return record.detail()

  def layout(self, project_id: str) -> LayoutResponse:
    record = self._records.get(project_id)
    if record is None:
      raise KeyError(project_id)
    return record.layout()

  def inventory(self, project_id: str) -> InventoryResponse:
    record = self._records.get(project_id)
    if record is None:
      raise KeyError(project_id)
    return record.inventory()

  def kpis(self, project_id: str) -> KpiSeries:
    record = self._records.get(project_id)
    if record is None:
      raise KeyError(project_id)
    return record.kpis()
