"""
Expose RAVEN entity "autocomplete" options for the PRLO XML builder.

This is intentionally lightweight and schema-adjacent: it prefers RAVEN's own
registered factories and input specifications over hard-coded lists.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET


def _safe_import(path: str):
  try:
    module = __import__(path, fromlist=["*"])
  except Exception:
    return None
  return module


def _project_root() -> Path:
  current = Path(__file__).resolve()
  for parent in [current.parent] + list(current.parents):
    if (parent / "ravenframework").is_dir():
      return parent
  return current.parents[1]


def _generated_xsd_dir() -> Path:
  return _project_root() / "developer_tools" / "XSDSchemas" / "generated"


def _xsd_path(entity: str) -> Optional[Path]:
  candidate = _generated_xsd_dir() / f"{entity}.xsd"
  if candidate.exists():
    return candidate
  fallback = _project_root() / "developer_tools" / "XSDSchemas" / f"{entity}.xsd"
  return fallback if fallback.exists() else None


def _xsd_root_type(xsd_root: ET.Element, entity: str) -> Optional[str]:
  for elem in xsd_root.iter():
    if elem.tag.endswith("element") and elem.attrib.get("name") == entity:
      return elem.attrib.get("type")
  return None


def _xsd_type_node(xsd_root: ET.Element, type_name: str) -> Optional[ET.Element]:
  for elem in xsd_root.iter():
    if elem.tag.endswith("complexType") and elem.attrib.get("name") == type_name:
      return elem
  return None


def _xsd_direct_children(type_node: ET.Element) -> List[Dict[str, str]]:
  options: List[Dict[str, str]] = []
  for container in list(type_node):
    if not (container.tag.endswith("sequence") or container.tag.endswith("choice")):
      continue
    for child in list(container):
      if not child.tag.endswith("element"):
        continue
      name = child.attrib.get("name")
      if not name:
        continue
      min_occurs = child.attrib.get("minOccurs", "1")
      max_occurs = child.attrib.get("maxOccurs", "1")
      options.append({
        "name": name,
        "type": child.attrib.get("type", ""),
        "minOccurs": min_occurs,
        "maxOccurs": max_occurs,
        "required": min_occurs != "0"
      })
  return options


def _xsd_required_attrs(xsd_root: ET.Element, type_name: str) -> List[str]:
  node = _xsd_type_node(xsd_root, type_name)
  if node is None:
    return []
  required: List[str] = []
  for attr in node.iter():
    if attr.tag.endswith("attribute") and attr.attrib.get("use") == "required":
      name = attr.attrib.get("name")
      if name and name not in required:
        required.append(name)
  return required


def _build_complete_template(
  xsd_root: ET.Element,
  tag: str,
  type_name: str,
  depth: int = 0,
  max_depth: int = 3
) -> str:
  """
  Recursively build a complete XML template with required child elements.

  Args:
    xsd_root: The XSD root element
    tag: The XML tag name
    type_name: The XSD type name
    depth: Current recursion depth
    max_depth: Maximum recursion depth to prevent infinite loops

  Returns:
    Complete XML template string with placeholders
  """
  if depth >= max_depth or not type_name:
    # At max depth or no type, return simple element
    return f"<{tag}><!-- TODO: Fill in --></{tag}>"

  indent = "  " * depth
  child_indent = "  " * (depth + 1)

  # Get required attributes
  required_attrs = _xsd_required_attrs(xsd_root, type_name)
  attrs = required_attrs[:] if required_attrs else []
  if "name" not in attrs and depth == 0:
    # Only add name attribute at top level
    attrs.insert(0, "name")

  # Build attribute string with placeholders
  attrs_text = ""
  if attrs:
    attr_parts = []
    for attr_name in attrs:
      if attr_name == "name":
        attr_parts.append(f'{attr_name}="TODO_NAME"')
      else:
        attr_parts.append(f'{attr_name}="TODO_{attr_name.upper()}"')
    attrs_text = " " + " ".join(attr_parts)

  # Get type node and children
  type_node = _xsd_type_node(xsd_root, type_name)
  if type_node is None:
    return f"{indent}<{tag}{attrs_text}><!-- TODO: Fill in --></{tag}>"

  children_info = _xsd_direct_children(type_node)
  required_children = [child for child in children_info if child.get("required", False)]

  if not required_children:
    # No required children, check if it's a simple type
    return f"{indent}<{tag}{attrs_text}><!-- TODO: Fill in --></{tag}>"

  # Build element with required children
  lines = [f"{indent}<{tag}{attrs_text}>"]

  for child in required_children:
    child_tag = child["name"]
    child_type = child["type"]

    # Add comment hint for the child
    lines.append(f"{child_indent}<!-- {child_tag}: Required -->")

    if child_type and depth + 1 < max_depth:
      # Recursively build child template
      child_template = _build_complete_template(
        xsd_root, child_tag, child_type, depth + 1, max_depth
      )
      lines.append(child_template)
    else:
      # Simple child element
      lines.append(f"{child_indent}<{child_tag}><!-- TODO: Fill in --></{child_tag}>")

  lines.append(f"{indent}</{tag}>")
  return "\n".join(lines)


def _build_template_from_attrs(tag: str, required_attrs: List[str]) -> str:
  attrs = required_attrs[:]
  if "name" not in attrs:
    attrs.insert(0, "name")
  attrs_text = " ".join(f'{name}=""' for name in attrs)
  return f"<{tag} {attrs_text}>\n</{tag}>"


@lru_cache(maxsize=128)
def _xsd_entity_options(entity: str) -> List[Dict[str, str]]:
  xsd_path = _xsd_path(entity)
  if xsd_path is None:
    return []
  try:
    tree = ET.parse(str(xsd_path))
  except ET.ParseError:
    return []
  root = tree.getroot()
  root_type = _xsd_root_type(root, entity)
  if root_type is None:
    return []
  type_node = _xsd_type_node(root, root_type)
  if type_node is None:
    return []
  options: List[Dict[str, str]] = []
  for item in _xsd_direct_children(type_node):
    tag = item["name"]
    type_name = item["type"]
    # Build complete template with required children
    if type_name:
      template = _build_complete_template(root, tag, type_name, depth=0, max_depth=3)
    else:
      # Fallback to simple template
      required_attrs = _xsd_required_attrs(root, type_name) if type_name else []
      template = _build_template_from_attrs(tag, required_attrs)
    options.append(
      {
        "tag": tag,
        "description": "",
        "template": template,
      }
    )
  return sorted(options, key=lambda opt: opt["tag"])


def _get_factory(entity: str):
  """
  Return a RAVEN factory object (EntityFactory) for the given entity name.
  """
  mapping = {
    "Samplers": "ravenframework.Samplers.Factory",
    "Optimizers": "ravenframework.Optimizers.Factory",
    "Models": "ravenframework.Models.Factory",
    "OutStreams": "ravenframework.OutStreams.Factory",
    "DataObjects": "ravenframework.DataObjects.Factory",
    "Databases": "ravenframework.Databases.Factory",
    "Distributions": "ravenframework.Distributions.Factory",
    "PostProcessors": "ravenframework.PostProcessors.Factory",
    "Metrics": "ravenframework.Metrics.Factory",
    "Steps": "ravenframework.Steps.Factory",
    "Files": "ravenframework.Files",
  }
  module_path = mapping.get(entity)
  if module_path is None:
    if entity == "Functions":
      module = _safe_import("ravenframework.Functions")
      if module is None:
        return None
      return getattr(module, "factory", None)
    return None
  module = _safe_import(module_path)
  if module is None:
    return None
  return getattr(module, "factory", None)

def _registered_types_from_factory_file(path: Path) -> List[str]:
  if not path.exists():
    return []
  text = path.read_text(encoding="utf-8", errors="ignore")
  types = re.findall(r"registerType\(\s*'([^']+)'\s*,", text)
  if not types:
    types = re.findall(r'registerType\(\s*\"([^\"]+)\"\s*,', text)
  # Preserve order but remove duplicates.
  seen = set()
  ordered: List[str] = []
  for name in types:
    if name in seen:
      continue
    seen.add(name)
    ordered.append(name)
  return ordered


def _types_from_factory_imports(path: Path, exclude: Optional[str] = None) -> List[str]:
  """
  Heuristic fallback for factories that use registerAllSubtypes(baseType).

  This pulls direct `from .X import ClassName` statements from the factory file and
  returns the imported class names (excluding the base type, if provided).
  """
  if not path.exists():
    return []
  text = path.read_text(encoding="utf-8", errors="ignore")
  imported = re.findall(r"from\s+\.[A-Za-z0-9_]+\s+import\s+([A-Za-z_][A-Za-z0-9_]*)", text)
  names: List[str] = []
  seen = set()
  for name in imported:
    if exclude and name == exclude:
      continue
    if name in seen:
      continue
    seen.add(name)
    names.append(name)
  return names


def _spec_description(entity_class) -> Optional[str]:
  """
  Attempt to extract a short description from the InputData spec.
  """
  getter = getattr(entity_class, "getInputSpecification", None)
  if getter is None:
    return None
  try:
    spec = getter()
  except Exception:
    return None
  return getattr(spec, "description", None)


def _build_entity_template(tag: str, entity_class=None) -> str:
  """
  Build a complete XML template from entity class InputSpecification.

  Includes required attributes and child elements.
  """
  required_attrs: List[str] = []
  required_children: List[str] = []

  if entity_class is not None:
    getter = getattr(entity_class, "getInputSpecification", None)
    if getter is not None:
      try:
        spec = getter()
        # Get required attributes (XML attributes)
        for name, info in getattr(spec, "parameters", {}).items():
          if info.get("required"):
            required_attrs.append(name)

        # Get required subnodes (child elements)
        subs = getattr(spec, "subs", None)
        if subs:
          for sub in subs:
            # Check if subnode is required
            sub_name = getattr(sub, "name", None)
            min_occurs = getattr(sub, "min", 0)
            if sub_name and min_occurs > 0:
              required_children.append(sub_name)
      except Exception:
        required_attrs = []
        required_children = []

  if not required_attrs:
    required_attrs = ["name"]

  # Build attribute string with placeholders
  attr_parts = []
  for attr_name in required_attrs:
    if attr_name == "name":
      attr_parts.append(f'{attr_name}="TODO_NAME"')
    else:
      attr_parts.append(f'{attr_name}="TODO_{attr_name.upper()}"')
  attrs = " ".join(attr_parts)

  # Build complete template with children
  if not required_children:
    return f"<{tag} {attrs}>\n  <!-- TODO: Fill in content -->\n</{tag}>"

  lines = [f"<{tag} {attrs}>"]
  for child_name in required_children:
    lines.append(f"  <!-- {child_name}: Required -->")
    lines.append(f"  <{child_name}><!-- TODO: Fill in --></{child_name}>")
  lines.append(f"</{tag}>")

  return "\n".join(lines)


def _xsd_type_options(
  entity: str,
  type_suffix: str = "SpecInputDataUser_type"
) -> List[Dict[str, str]]:
  """
  Look up options for a specific XSD type across all entity XSD files.

  This handles sub-entities like "MultiRun" which are types within Steps.xsd.
  """
  # Try to find the type in various XSD files
  xsd_files = list(_generated_xsd_dir().glob("*.xsd"))
  type_name = f"{entity}{type_suffix}"

  for xsd_path in xsd_files:
    try:
      tree = ET.parse(str(xsd_path))
      root = tree.getroot()

      # Look for the type definition
      type_node = _xsd_type_node(root, type_name)
      if type_node is None:
        continue

      # Found the type! Extract its children
      options: List[Dict[str, str]] = []
      for item in _xsd_direct_children(type_node):
        tag = item["name"]
        item_type = item["type"]
        # Build template
        if item_type:
          template = _build_complete_template(
            root, tag, item_type, depth=0, max_depth=3
          )
        else:
          required_attrs = (
            _xsd_required_attrs(root, item_type) if item_type else []
          )
          template = _build_template_from_attrs(tag, required_attrs)
        options.append({
          "tag": tag,
          "description": "",
          "template": template,
        })
      return sorted(options, key=lambda opt: opt["tag"])
    except ET.ParseError:
      continue

  return []


@lru_cache(maxsize=256)
def entity_options(entity: str) -> List[Dict[str, str]]:
  """
  Return autocomplete options for a given RAVEN XML entity or sub-entity.

  Each option includes:
  - tag: the XML tag to insert
  - description: short help text (best-effort)
  - template: insertable XML snippet
  """
  xsd_options = _xsd_entity_options(entity)
  if xsd_options:
    return xsd_options

  # Try as a sub-entity type (e.g., "MultiRun" within Steps.xsd)
  type_options = _xsd_type_options(entity)
  if type_options:
    return type_options

  # Fallback: read Factory.py without importing modules (avoids optional deps).
  factory_files = {
    "Samplers": _project_root() / "ravenframework" / "Samplers" / "Factory.py",
    "Optimizers": _project_root() / "ravenframework" / "Optimizers" / "Factory.py",
    "Models": _project_root() / "ravenframework" / "Models" / "Factory.py",
    "OutStreams": _project_root() / "ravenframework" / "OutStreams" / "Factory.py",
    "DataObjects": _project_root() / "ravenframework" / "DataObjects" / "Factory.py",
    "Databases": _project_root() / "ravenframework" / "Databases" / "Factory.py",
    "Distributions": _project_root() / "ravenframework" / "Distributions" / "Factory.py",
    "PostProcessors": _project_root() / "ravenframework" / "PostProcessors" / "Factory.py",
    "Metrics": _project_root() / "ravenframework" / "Metrics" / "Factory.py",
    "Steps": _project_root() / "ravenframework" / "Steps" / "Factory.py",
  }
  factory_file = factory_files.get(entity)
  if factory_file is not None:
    explicit = _registered_types_from_factory_file(factory_file)
    if explicit:
      return [
        {
          "tag": type_name,
          "description": "",
          "template": _build_entity_template(type_name),
        }
        for type_name in sorted(explicit)
      ]
    # Factories like Models/Steps use registerAllSubtypes; fall back to imports.
    exclude = "Model" if entity == "Models" else ("Step" if entity == "Steps" else None)
    inferred = _types_from_factory_imports(factory_file, exclude=exclude)
    return [
      {"tag": type_name, "description": "", "template": _build_entity_template(type_name)}
      for type_name in inferred
    ]

  # Small factories defined in single modules (avoid imports).
  if entity == "Functions":
    func_file = _project_root() / "ravenframework" / "Functions.py"
    return [
      {"tag": type_name, "description": "", "template": _build_entity_template(type_name)}
      for type_name in sorted(_registered_types_from_factory_file(func_file))
    ]

  if entity == "Files":
    files_file = _project_root() / "ravenframework" / "Files.py"
    return [
      {"tag": type_name, "description": "", "template": _build_entity_template(type_name)}
      for type_name in sorted(_registered_types_from_factory_file(files_file))
    ]

  # As a last resort, attempt to import the factory (may require optional deps).
  factory = _get_factory(entity)
  if factory is not None:
    options: List[Dict[str, str]] = []
    for type_name in sorted(factory.knownTypes()):
      try:
        cls = factory.returnClass(type_name)
      except Exception:
        cls = None
      options.append(
        {
          "tag": type_name,
          "description": (_spec_description(cls) or ""),
          "template": _build_entity_template(type_name, cls),
        }
      )
    return options

  if entity == "VariableGroups":
    return [
      {
        "tag": "Group",
        "description": "Defines a named group of variables for reuse throughout the input.",
        "template": '<Group name="">var1,var2</Group>',
      }
    ]

  if entity == "TestInfo":
    return [
      {"tag": "name", "description": "Test name.", "template": "<name></name>"},
      {"tag": "author", "description": "Author.", "template": "<author></author>"},
      {"tag": "created", "description": "Creation date.", "template": "<created></created>"},
      {"tag": "classesTested", "description": "Classes tested.", "template": "<classesTested></classesTested>"},
      {"tag": "description", "description": "Freeform description.", "template": "<description>\n</description>"},
    ]

  if entity == "RunInfo":
    return [
      {
        "tag": tag,
        "description": "",
        "template": f"<{tag}></{tag}>",
      }
      for tag in _runinfo_tags()
    ]

  # No known options for this entity.
  return []


@lru_cache(maxsize=1)
def _runinfo_tags() -> List[str]:
  """
  Extract supported <RunInfo> children from `ravenframework/Simulation.py`.
  """
  sim_path = _project_root() / "ravenframework" / "Simulation.py"
  if not sim_path.exists():
    return []
  text = sim_path.read_text(encoding="utf-8", errors="ignore")
  # Find `elif element.tag == 'X':` occurrences.
  tags = set(re.findall(r"elif\s+element\.tag\s*==\s*'([^']+)'\s*:", text))
  # Include `if element.tag ==` as well.
  tags.update(re.findall(r"if\s+element\.tag\s*==\s*'([^']+)'\s*:", text))
  # Filter out internal or legacy checks.
  tags.discard("printInput")  # keep? still useful, but noisy; users can still add manually.
  # Stable sort.
  return sorted(tags)
