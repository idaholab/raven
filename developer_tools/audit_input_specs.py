#!/usr/bin/env python3
"""Audit InputData spec coverage and XSD alignment in RAVEN.

Outputs a text report to stdout and optionally writes JSON/markdown if requested.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_XSD_DIR = REPO_ROOT / "developer_tools" / "XSDSchemas" / "generated"

ENTITY_FACTORIES: Dict[str, Path] = {
    "Samplers": REPO_ROOT / "ravenframework" / "Samplers" / "Factory.py",
    "Optimizers": REPO_ROOT / "ravenframework" / "Optimizers" / "Factory.py",
    "Models": REPO_ROOT / "ravenframework" / "Models" / "Factory.py",
    "OutStreams": REPO_ROOT / "ravenframework" / "OutStreams" / "Factory.py",
    "DataObjects": REPO_ROOT / "ravenframework" / "DataObjects" / "Factory.py",
    "Databases": REPO_ROOT / "ravenframework" / "Databases" / "Factory.py",
    "Distributions": REPO_ROOT / "ravenframework" / "Distributions.py",
    "PostProcessors": REPO_ROOT / "ravenframework" / "PostProcessors" / "Factory.py",
    "Metrics": REPO_ROOT / "ravenframework" / "Metrics" / "Factory.py",
    "Steps": REPO_ROOT / "ravenframework" / "Steps" / "Factory.py",
    # Small factories defined in single modules.
    "Functions": REPO_ROOT / "ravenframework" / "Functions.py",
    "Files": REPO_ROOT / "ravenframework" / "Files.py",
}

XSD_FILES: Dict[str, Path] = {
    "Samplers": REPO_ROOT / "developer_tools" / "XSDSchemas" / "Samplers.xsd",
    "Optimizers": REPO_ROOT / "developer_tools" / "XSDSchemas" / "Optimizers.xsd",
    "Models": REPO_ROOT / "developer_tools" / "XSDSchemas" / "Models.xsd",
    "OutStreams": REPO_ROOT / "developer_tools" / "XSDSchemas" / "OutstreamManager.xsd",
    "DataObjects": REPO_ROOT / "developer_tools" / "XSDSchemas" / "DataObjects.xsd",
    "Databases": REPO_ROOT / "developer_tools" / "XSDSchemas" / "Databases.xsd",
    "Distributions": REPO_ROOT / "developer_tools" / "XSDSchemas" / "Distributions.xsd",
    "Metrics": REPO_ROOT / "developer_tools" / "XSDSchemas" / "Metrics.xsd",
    "Steps": REPO_ROOT / "developer_tools" / "XSDSchemas" / "Steps.xsd",
    "Functions": REPO_ROOT / "developer_tools" / "XSDSchemas" / "Functions.xsd",
    "Files": REPO_ROOT / "developer_tools" / "XSDSchemas" / "Files.xsd",
    "VariableGroups": REPO_ROOT / "developer_tools" / "XSDSchemas" / "VarGroups.xsd",
    "TestInfo": REPO_ROOT / "developer_tools" / "XSDSchemas" / "TestInfo.xsd",
    "OutStreamsManager": REPO_ROOT / "developer_tools" / "XSDSchemas" / "OutstreamManager.xsd",
}

XSD_ROOT_TYPES: Dict[str, str] = {
    "Samplers": "SamplerData",
    "Optimizers": "OptimizerData",
    "Models": "ModelsData",
    "OutStreams": "OutStreamData",
    "DataObjects": "DataObjectsData",
    "Databases": "DatabaseType",
    "Distributions": "DistributionData",
    "Metrics": "MetricsData",
    "Steps": "StepType",
    "Functions": "FunctionType",
    "Files": "FilesType",
    "VariableGroups": "VarGroupsType",
    "TestInfo": "TestInfoData",
}

REGISTER_RE = re.compile(r"registerType\(\s*['\"]([^'\"]+)['\"]\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)")
FROM_IMPORT_RE = re.compile(r"^from\s+\.([A-Za-z0-9_]+)\s+import\s+([A-Za-z0-9_,\s]+)$", re.M)
CLASS_RE = re.compile(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\b")

XML_PARSE_HINTS = (
    "XMLread",
    "_readMoreXML",
    "handleInput",
    "xml.etree",
    "ElementTree",
    "xmlUtils",
    ".find(",
    ".findall(",
)


@dataclass
class ClassAudit:
    class_name: str
    module_path: Optional[Path]
    has_get_input_spec: bool
    has_xmlread: bool
    has_read_more_xml: bool
    has_handle_input: bool


@dataclass
class EntityAudit:
    entity: str
    types: List[str]
    missing_spec: List[str]
    unresolved: List[str]
    errors: List[str]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def _parse_factory_types(factory_path: Path) -> Tuple[List[Tuple[str, str]], Dict[str, str], Dict[str, str]]:
    """Return [(type_name, class_name)], mapping of class_name->module_name, and alias->original."""
    text = _read_text(factory_path)
    pairs = REGISTER_RE.findall(text)
    imports: Dict[str, str] = {}
    aliases: Dict[str, str] = {}
    for mod, names_blob in FROM_IMPORT_RE.findall(text):
        names = [n.strip() for n in names_blob.split(",") if n.strip()]
        for name in names:
            if " as " in name:
                original, alias = name.split(" as ", 1)
                original = original.strip()
                alias = alias.strip()
                imports[alias] = mod
                aliases[alias] = original
                continue
            imports[name] = mod
    return pairs, imports, aliases


def _find_class_file(search_root: Path, class_name: str) -> Optional[Path]:
    for path in search_root.rglob("*.py"):
        if path.name.startswith("_"):
            continue
        info = _class_info(path)
        if class_name in info:
            return path
    return None


def _class_methods(path: Path, class_name: str) -> Set[str]:
    text = _read_text(path)
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            return methods
    return set()


_CLASS_INFO_CACHE: Dict[Path, Dict[str, Tuple[Set[str], List[str]]]] = {}


def _class_info(path: Path) -> Dict[str, Tuple[Set[str], List[str]]]:
    cached = _CLASS_INFO_CACHE.get(path)
    if cached is not None:
        return cached
    text = _read_text(path)
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        _CLASS_INFO_CACHE[path] = {}
        return {}
    info: Dict[str, Tuple[Set[str], List[str]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
        bases: List[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
        info[node.name] = (methods, bases)
    _CLASS_INFO_CACHE[path] = info
    return info


def _class_has_method(
    path: Path,
    class_name: str,
    method_name: str,
    search_root: Path,
    seen: Optional[Set[Tuple[Path, str]]] = None,
) -> bool:
    info = _class_info(path)
    if class_name not in info:
        return False
    methods, bases = info[class_name]
    if method_name in methods:
        return True
    if seen is None:
        seen = set()
    seen_key = (path, class_name)
    if seen_key in seen:
        return False
    seen.add(seen_key)
    for base in bases:
        if base in info:
            if _class_has_method(path, base, method_name, search_root, seen):
                return True
            continue
        base_path = _find_class_file(search_root, base)
        if base_path and _class_has_method(base_path, base, method_name, search_root, seen):
            return True
    return False

def _find_register_all_subtypes(factory_path: Path) -> List[str]:
    text = _read_text(factory_path)
    try:
        tree = ast.parse(text, filename=str(factory_path))
    except SyntaxError:
        return []
    base_names: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "registerAllSubtypes":
            if node.args and isinstance(node.args[0], ast.Name):
                base_names.append(node.args[0].id)
    return base_names


def _find_subclasses(search_paths: Iterable[Path], base_names: Set[str]) -> Set[str]:
    class_bases: Dict[str, Set[str]] = {}
    for path in search_paths:
        if path.is_dir():
            paths = [p for p in path.rglob("*.py") if p.is_file()]
        else:
            paths = [path]
        for file_path in paths:
            info = _class_info(file_path)
            for class_name, (_, bases) in info.items():
                class_bases.setdefault(class_name, set()).update(bases)
    initial = set(base_names)
    added = True
    while added:
        added = False
        for class_name, bases in class_bases.items():
            if class_name in base_names:
                continue
            if any(base in base_names for base in bases):
                base_names.add(class_name)
                added = True
    return base_names - initial


def _audit_entity(entity: str) -> EntityAudit:
    factory_path = ENTITY_FACTORIES.get(entity)
    if factory_path is None or not factory_path.exists():
        return EntityAudit(entity, [], [], [], ["factory not found"])
    pairs, imports, aliases = _parse_factory_types(factory_path)
    register_all = _find_register_all_subtypes(factory_path)
    # If no explicit registerType calls, fall back to imported classes as a weak proxy.
    if not pairs and not register_all:
        for cls_name, mod in imports.items():
            pairs.append((cls_name, cls_name))
    if register_all:
        base_names: Set[str] = set()
        for base in register_all:
            alias_target = aliases.get(base)
            if alias_target:
                base_names.add(alias_target)
            else:
                base_names.add(base)
        if factory_path.name == "Factory.py":
            search_root = factory_path.parent
        else:
            search_root = factory_path
        subtypes = _find_subclasses([search_root], base_names)
        for subtype in sorted(subtypes):
            pairs.append((subtype, subtype))
    types: List[str] = []
    missing_spec: List[str] = []
    unresolved: List[str] = []
    errors: List[str] = []

    # Set the search root to the entity folder or ravenframework.
    if factory_path.name == "Factory.py":
        search_root = factory_path.parent
    else:
        search_root = REPO_ROOT / "ravenframework"

    for type_name, class_name in pairs:
        if type_name not in types:
            types.append(type_name)
        module_name = imports.get(class_name)
        module_path: Optional[Path] = None
        if module_name:
            module_path = factory_path.parent / f"{module_name}.py"
        if module_path is None or not module_path.exists():
            module_path = _find_class_file(search_root, class_name)
        if module_path is not None:
            if class_name not in _class_info(module_path):
                module_path = _find_class_file(search_root, class_name)
        if module_path is None:
            unresolved.append(type_name)
            continue
        if not _class_has_method(module_path, class_name, "getInputSpecification", search_root):
            missing_spec.append(type_name)
    return EntityAudit(entity, sorted(types), sorted(missing_spec), sorted(set(unresolved)), errors)


def _scan_manual_parsing(paths: Iterable[Path]) -> Dict[str, List[str]]:
    """Scan for XML parsing hints in classes missing getInputSpecification."""
    result: Dict[str, List[str]] = {}
    for path in paths:
        text = _read_text(path)
        if not text:
            continue
        if not any(hint in text for hint in XML_PARSE_HINTS):
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            if "getInputSpecification" in methods:
                continue
            has_hint = (
                "XMLread" in methods
                or "_readMoreXML" in methods
                or "handleInput" in methods
            )
            if has_hint:
                result.setdefault(str(path), []).append(node.name)
    return result


def _collect_py_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.py") if p.is_file()]


def _xsd_type_names(xsd_path: Path, root_type: Optional[str] = None) -> Set[str]:
    if not xsd_path.exists():
        return set()
    try:
        tree = ET.parse(str(xsd_path))
    except ET.ParseError:
        return set()
    root = tree.getroot()
    if root_type is None:
        names: Set[str] = set()
        for elem in root.iter():
            if elem.tag.endswith("element"):
                name = elem.attrib.get("name")
                if name:
                    names.add(name)
        return names
    for ctype in root.iter():
        if ctype.tag.endswith("complexType") and ctype.attrib.get("name") == root_type:
            names: Set[str] = set()
            for elem in ctype.iter():
                if elem.tag.endswith("element"):
                    name = elem.attrib.get("name")
                    if name:
                        names.add(name)
            return names
    return set()

def _xsd_root_type(xsd_path: Path, entity: str, fallback: Optional[str]) -> Optional[str]:
    if not xsd_path.exists():
        return fallback
    try:
        tree = ET.parse(str(xsd_path))
    except ET.ParseError:
        return fallback
    root = tree.getroot()
    for elem in root.iter():
        if elem.tag.endswith("element") and elem.attrib.get("name") == entity:
            return elem.attrib.get("type", fallback)
    return fallback

def _resolve_xsd_path(entity: str) -> Optional[Path]:
    generated = GENERATED_XSD_DIR / f"{entity}.xsd"
    if generated.exists():
        return generated
    return XSD_FILES.get(entity)


def run_audit() -> Dict[str, object]:
    entities: List[EntityAudit] = []
    for entity in sorted(ENTITY_FACTORIES.keys()):
        entities.append(_audit_entity(entity))

    # Manual XML parsing scan
    raven_files = _collect_py_files(REPO_ROOT / "ravenframework")
    plugin_files = _collect_py_files(REPO_ROOT / "plugins")
    manual_raven = _scan_manual_parsing(raven_files)
    manual_plugins = _scan_manual_parsing(plugin_files)

    # XSD vs factory type check (approximate)
    xsd_diff: Dict[str, Dict[str, List[str]]] = {}
    for audit in entities:
        xsd_path = _resolve_xsd_path(audit.entity)
        if not xsd_path:
            continue
        root_type = _xsd_root_type(xsd_path, audit.entity, XSD_ROOT_TYPES.get(audit.entity))
        xsd_names = _xsd_type_names(xsd_path, root_type=root_type)
        type_names = set(audit.types)
        if not xsd_names:
            continue
        if audit.entity == "Steps":
            type_names.discard("Step")
        if audit.entity == "Models":
            type_names.discard("Model")
        for name in list(type_names):
            if "-" in name and name.replace("-", "") in type_names:
                type_names.discard(name)
        missing_in_xsd = sorted(type_names - xsd_names)
        extra_in_xsd = sorted(xsd_names - type_names)
        if audit.entity == "Models":
            extra_in_xsd = [name for name in extra_in_xsd if name != "HybridModelBase"]
        if audit.entity == "Distributions":
            missing_in_xsd = [name for name in missing_in_xsd if name != "BoostDistribution"]
        xsd_diff[audit.entity] = {
            "missing_in_xsd": missing_in_xsd,
            "extra_in_xsd": extra_in_xsd,
        }

    return {
        "entities": [audit.__dict__ for audit in entities],
        "manual_parsing": {
            "ravenframework": manual_raven,
            "plugins": manual_plugins,
        },
        "xsd_diff": xsd_diff,
    }


def _print_report(data: Dict[str, object]) -> None:
    entities = data.get("entities", [])
    print("InputData Spec Coverage Audit")
    print("=" * 32)
    for ent in entities:
        entity = ent["entity"]
        missing = ent["missing_spec"]
        unresolved = ent["unresolved"]
        if not missing and not unresolved:
            continue
        print(f"\n[{entity}]")
        if missing:
            print(f"  Missing getInputSpecification: {len(missing)}")
            for name in missing[:50]:
                print(f"    - {name}")
            if len(missing) > 50:
                print("    ...")
        if unresolved:
            print(f"  Unresolved types: {len(unresolved)}")
            for name in unresolved[:50]:
                print(f"    - {name}")
            if len(unresolved) > 50:
                print("    ...")

    manual = data.get("manual_parsing", {})
    print("\nManual XML Parsing (classes without getInputSpecification)")
    print("=" * 58)
    for scope in ("ravenframework", "plugins"):
        items = manual.get(scope, {})
        print(f"\n[{scope}] {len(items)} files")
        shown = 0
        for path, classes in items.items():
            print(f"  {path}")
            for cls in classes:
                print(f"    - {cls}")
            shown += 1
            if shown >= 50:
                print("  ...")
                break

    xsd_diff = data.get("xsd_diff", {})
    print("\nXSD vs Factory Type Names (approximate)")
    print("=" * 46)
    for entity, diff in xsd_diff.items():
        missing = diff.get("missing_in_xsd", [])
        extra = diff.get("extra_in_xsd", [])
        if not missing and not extra:
            continue
        print(f"\n[{entity}]")
        if missing:
            print(f"  In factory, missing in XSD: {len(missing)}")
            for name in missing[:50]:
                print(f"    - {name}")
            if len(missing) > 50:
                print("    ...")
        if extra:
            print(f"  In XSD, missing in factory: {len(extra)}")
            for name in extra[:50]:
                print(f"    - {name}")
            if len(extra) > 50:
                print("    ...")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", dest="json_path", help="Write JSON report to path")
    parser.add_argument("--no-print", action="store_true", help="Skip text report")
    args = parser.parse_args()

    data = run_audit()
    if not args.no_print:
        _print_report(data)
    if args.json_path:
        out = Path(args.json_path)
        out.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
