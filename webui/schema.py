"""
Pydantic data models for the PRLO web UI backend.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

try:
  from pydantic import BaseModel, Field
except ModuleNotFoundError:  # pragma: no cover
  class _FieldPlaceholder:
    def __init__(self, default=None, default_factory=None):
      self.default = default
      self.default_factory = default_factory

    def materialize(self):
      if self.default_factory is not None:
        return self.default_factory()
      return self.default

  def Field(default=None, default_factory=None, **_kwargs):  # type: ignore[misc]
    return _FieldPlaceholder(default=default, default_factory=default_factory)

  class BaseModel:  # type: ignore[override]
    """
    Lightweight drop-in replacement when pydantic is unavailable.
    """

    def __init__(self, **data):
      defaults = self._gather_defaults()
      for key, placeholder in defaults.items():
        if key not in data:
          data[key] = placeholder.materialize()
      for key, value in data.items():
        setattr(self, key, value)

    @classmethod
    def _gather_defaults(cls):
      defaults = {}
      for name, value in cls.__dict__.items():
        if isinstance(value, _FieldPlaceholder):
          defaults[name] = value
      return defaults

    def dict(self) -> Dict[str, object]:
      return dict(self.__dict__)


class Health(BaseModel):
  """Simple health response."""

  status: str = Field(default="ok")


class ProjectSummary(BaseModel):
  """Minimal information about a configured project."""

  id: str = Field(description="Stable slug used in URLs.")
  name: str = Field(description="Human-friendly label.")
  path: str = Field(description="Absolute filesystem location.")
  core_type: Optional[str] = Field(default=None)
  symmetry: Optional[str] = Field(default=None)


class ProjectDetail(ProjectSummary):
  """Extended project metadata."""

  description: Optional[str] = Field(default=None)
  updated_at: Optional[datetime] = Field(default=None)


class AssemblyType(BaseModel):
  """Defines a distinct assembly profile."""

  id: int
  name: str
  serial_label: Optional[str] = Field(default=None)
  assembly_type: Optional[str] = Field(default=None, alias="type")
  enrichment_wt: Optional[float] = Field(default=None)
  burnup_status: Optional[str] = Field(default=None)
  attributes: Dict[str, str] = Field(default_factory=dict)


class InventoryItem(BaseModel):
  """Inventory entry referencing an assembly type."""

  id: int
  assembly_type_id: int = Field(description="Reference to AssemblyType.id")
  quantity: int
  status: str = Field(default="available")
  reserved: int = Field(default=0)
  notes: Optional[str] = Field(default=None)


class SlotDefinition(BaseModel):
  """Describes a logical slot (raven variable) in the core."""

  slot_id: int = Field(description="Zero-based slot identifier.")
  variable: str = Field(description="RAVEN decision variable (e.g. loc5).")
  template_label: str = Field(description="SIMULATE placeholder (e.g. $fa-loc5$).")
  symmetry_positions: List[str] = Field(
    default_factory=list,
    description="Canonical lattice coordinates associated with the slot.",
  )


class LayoutPlacement(BaseModel):
  """Maps a slot to an assembly assignment."""

  slot_id: int
  assembly_id: Optional[int] = Field(default=None)
  label: Optional[str] = Field(default=None, description="Renderer hint or alias.")
  source: Optional[str] = Field(default=None, description="Data provenance.")


class LayoutResponse(BaseModel):
  """Payload returned for layout requests."""

  project_id: str
  symmetry: Optional[str] = Field(default=None)
  slots: List[SlotDefinition]
  placements: List[LayoutPlacement]
  generated_at: datetime = Field(default_factory=datetime.utcnow)


class InventoryResponse(BaseModel):
  """Inventory payload for a project."""

  project_id: str
  assembly_types: List[AssemblyType]
  items: List[InventoryItem]
  generated_at: datetime = Field(default_factory=datetime.utcnow)


class KpiSample(BaseModel):
  """Optimization KPI record."""

  batch_id: Optional[int] = Field(default=None)
  iteration: Optional[int] = Field(default=None)
  rank: Optional[int] = Field(default=None)
  objective_values: Dict[str, float] = Field(default_factory=dict)
  constraints: Dict[str, float] = Field(default_factory=dict)
  metadata: Dict[str, str] = Field(default_factory=dict)


class KpiSeries(BaseModel):
  """Container for KPI time series."""

  project_id: str
  samples: List[KpiSample]
  generated_at: datetime = Field(default_factory=datetime.utcnow)


class LayoutUpdate(BaseModel):
  """Incoming layout update request."""

  placements: List[LayoutPlacement]
  note: Optional[str] = Field(default=None)
