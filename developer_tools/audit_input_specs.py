#!/usr/bin/env python3
# Copyright 2026 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Audit InputData spec coverage and XSD alignment in RAVEN.

Discovery is done by LIVE REGISTRY INTROSPECTION rather than by scraping the
``Factory.py`` sources with regular expressions.  For each entity we import the
real factory, enumerate ``factory.knownTypes()`` and resolve every registered
type with ``factory.returnClass()``.  Coverage of ``getInputSpecification`` is
then decided with a live ``hasattr``/MRO check (the authoritative answer),
falling back to an AST scan only when a class cannot be resolved live.

Any registered class that itself exposes a secondary registry dict (for example
``knownAlgorithms`` on the multi-objective genetic-algorithm optimizer) has its
sub-registered names folded into the audit as well, so algorithms registered via
``registerAlgorithm(...)`` are not invisible to the guardrail.

The module exposes two public entry points:

``run_audit()``
    The full, human-oriented audit dict (entities + manual-parsing scan +
    xsd_diff).  Used by the text report and the ``--json`` dump.

``coverage_facts(audit=None)``
    The minimal, *portable* regression baseline: per entity only the stable
    coverage facts (set of classes missing ``getInputSpecification``, the
    xsd_diff sets, and import/resolve errors).  It deliberately omits the
    volatile ``types`` rosters and the absolute-path manual-parsing map so that
    adding a well-formed entity does not churn the committed baseline, while a
    regressed spec / broken XSD coverage / newly introduced manual-parse class
    still trips the guard.

Outputs a text report to stdout and optionally writes JSON if requested.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_XSD_DIR = REPO_ROOT / "developer_tools" / "XSDSchemas" / "generated"

# Make sure ravenframework is importable when this module is run directly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Entity -> (importable factory module, attribute holding the EntityFactory).
# Discovery is live, so these point at the real factory objects, not source
# files to be scraped.  PostProcessors now resolves correctly (it was invisible
# to the old regex because its registrations live under Models/PostProcessors).
ENTITY_FACTORIES: Dict[str, Tuple[str, str]] = {
    "DataObjects": ("ravenframework.DataObjects.Factory", "factory"),
    "Databases": ("ravenframework.Databases.Factory", "factory"),
    "Distributions": ("ravenframework.Distributions", "factory"),
    "Files": ("ravenframework.Files", "factory"),
    "Functions": ("ravenframework.Functions", "factory"),
    "Metrics": ("ravenframework.Metrics.Factory", "factory"),
    "Models": ("ravenframework.Models.Factory", "factory"),
    "Optimizers": ("ravenframework.Optimizers.Factory", "factory"),
    "OutStreams": ("ravenframework.OutStreams.Factory", "factory"),
    "PostProcessors": ("ravenframework.Models.PostProcessors.Factory", "factory"),
    "Samplers": ("ravenframework.Samplers.Factory", "factory"),
    "Steps": ("ravenframework.Steps.Factory", "factory"),
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

# Attribute names on a registered class that themselves hold a secondary
# name->class registry (e.g. the MultiObjectiveGeneticAlgorithm exposes
# ``knownAlgorithms`` for NSGA-II / NSGA-III).  Generalize here so new
# sub-registries only need to be named, not special-cased.
SUBREGISTRY_ATTRS: Tuple[str, ...] = ("knownAlgorithms",)

# Method that marks a class as participating in the InputData spec system.
SPEC_METHOD = "getInputSpecification"

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
class EntityAudit:
    """Audit result for a single entity, from live-registry discovery."""
    entity: str
    types: List[str] = field(default_factory=list)
    missing_spec: List[str] = field(default_factory=list)
    unresolved: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Environment setup for standalone factory imports.
# --------------------------------------------------------------------------- #
def _prepare_import_environment() -> None:
    """Mirror the RAVEN driver's a-la-carte setup needed for bare imports.

    Some framework modules are decorated with ``@profile`` (the line_profiler
    builtin), which only exists when running under ``kernprof``.  The real
    driver installs a passthrough via ``DriverUtils.setupBuiltins``; replicate
    that here so entities such as ``Steps`` import cleanly outside a profiler.
    Also suppress the noisy per-spec InputData warnings.
    """
    try:
        from ravenframework.CustomDrivers import DriverUtils
        DriverUtils.setupBuiltins()
    except Exception:
        # Fall back to the minimal builtin shim if DriverUtils is unavailable.
        import builtins
        if not hasattr(builtins, "profile"):
            builtins.profile = lambda f: f
    try:
        from ravenframework.utils import InputData
        InputData.SUPPRESS_INPUT_SPEC_WARNINGS = True
    except Exception:
        pass


def _load_factory(module_path: str, attr: str):
    """Import ``module_path`` and return its factory attribute."""
    module = importlib.import_module(module_path)
    return getattr(module, attr)


# --------------------------------------------------------------------------- #
# AST fallback: only used when a class cannot be resolved live.
# --------------------------------------------------------------------------- #
_CLASS_INFO_CACHE: Dict[Path, Dict[str, Tuple[Set[str], List[str]]]] = {}


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


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


def _find_class_file(search_root: Path, class_name: str) -> Optional[Path]:
    for path in search_root.rglob("*.py"):
        if path.name.startswith("_"):
            continue
        if class_name in _class_info(path):
            return path
    return None


def _ast_class_has_method(
    path: Path,
    class_name: str,
    method_name: str,
    search_root: Path,
    seen: Optional[Set[Tuple[Path, str]]] = None,
) -> bool:
    """AST-only MRO walk; used solely as a fallback for unresolved classes."""
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
            if _ast_class_has_method(path, base, method_name, search_root, seen):
                return True
            continue
        base_path = _find_class_file(search_root, base)
        if base_path and _ast_class_has_method(base_path, base, method_name, search_root, seen):
            return True
    return False


# --------------------------------------------------------------------------- #
# Live discovery.
# --------------------------------------------------------------------------- #
def _live_has_spec(cls) -> bool:
    """Authoritative coverage check: does ``cls`` (or any ancestor) define the spec?

    ``hasattr`` is MRO-aware, so an inherited ``getInputSpecification`` counts as
    coverage, which is the correct semantics for RAVEN's class hierarchy.  It is
    ``False`` for a class that genuinely never provides the method (e.g. a
    manual-parse class), which is exactly what the guard needs to catch.
    """
    return hasattr(cls, SPEC_METHOD)


def _discover_entity_classes(factory) -> List[Tuple[str, object, Optional[str]]]:
    """Return [(type_name, class_or_None, resolve_error_or_None)] for a factory.

    Enumerates ``knownTypes`` plus any secondary ``knownAlgorithms``-style
    registry exposed by a resolved class, deterministically sorted.
    """
    discovered: Dict[str, Tuple[object, Optional[str]]] = {}
    for type_name in sorted(factory.knownTypes()):
        try:
            cls = factory.returnClass(type_name)
        except Exception as exc:  # pylint: disable=broad-except
            discovered[type_name] = (None, f"{type(exc).__name__}: {exc}")
            continue
        discovered[type_name] = (cls, None)
        # Fold in any secondary registry (e.g. registerAlgorithm sub-variants).
        for attr in SUBREGISTRY_ATTRS:
            subreg = getattr(cls, attr, None)
            if not isinstance(subreg, dict):
                continue
            for sub_name, sub_cls in subreg.items():
                if sub_name in discovered:
                    continue
                discovered[sub_name] = (sub_cls, None)
    return [(name, discovered[name][0], discovered[name][1]) for name in sorted(discovered)]


def _audit_entity(entity: str, module_path: str, attr: str) -> EntityAudit:
    audit = EntityAudit(entity)
    try:
        factory = _load_factory(module_path, attr)
    except Exception as exc:  # pylint: disable=broad-except
        audit.errors.append(f"failed to import factory: {type(exc).__name__}: {exc}")
        return audit

    search_root = REPO_ROOT / "ravenframework"
    types: Set[str] = set()
    missing_spec: Set[str] = set()
    unresolved: Set[str] = set()
    errors: List[str] = []

    for type_name, cls, resolve_err in _discover_entity_classes(factory):
        types.add(type_name)
        if cls is None:
            # Could not resolve the class live; try an AST fallback by name.
            unresolved.add(type_name)
            errors.append(f"resolve {type_name}: {resolve_err}")
            ast_path = _find_class_file(search_root, type_name)
            if ast_path is not None and not _ast_class_has_method(
                ast_path, type_name, SPEC_METHOD, search_root
            ):
                missing_spec.add(type_name)
            continue
        if not _live_has_spec(cls):
            missing_spec.add(type_name)

    audit.types = sorted(types)
    audit.missing_spec = sorted(missing_spec)
    audit.unresolved = sorted(unresolved)
    audit.errors = sorted(errors)
    return audit


# --------------------------------------------------------------------------- #
# Manual-parsing scan (human-report only; excluded from the portable baseline).
# --------------------------------------------------------------------------- #
def _scan_manual_parsing(paths: Iterable[Path]) -> Dict[str, List[str]]:
    """Scan for XML parsing hints in classes missing getInputSpecification.

    Keyed by repo-relative path so the human report is reproducible across
    checkouts; this map is intentionally NOT part of the regression baseline.
    """
    result: Dict[str, List[str]] = {}
    for path in paths:
        text = _read_text(path)
        if not text or not any(hint in text for hint in XML_PARSE_HINTS):
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        try:
            rel = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel = str(path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
            if SPEC_METHOD in methods:
                continue
            if "XMLread" in methods or "_readMoreXML" in methods or "handleInput" in methods:
                result.setdefault(rel, []).append(node.name)
    return {key: sorted(val) for key, val in sorted(result.items())}


def _collect_py_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


# --------------------------------------------------------------------------- #
# XSD alignment.
# --------------------------------------------------------------------------- #
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
            names = set()
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


def _compute_xsd_diff(entities: List[EntityAudit]) -> Dict[str, Dict[str, List[str]]]:
    xsd_diff: Dict[str, Dict[str, List[str]]] = {}
    for audit in entities:
        xsd_path = _resolve_xsd_path(audit.entity)
        if not xsd_path:
            continue
        root_type = _xsd_root_type(xsd_path, audit.entity, XSD_ROOT_TYPES.get(audit.entity))
        xsd_names = _xsd_type_names(xsd_path, root_type=root_type)
        if not xsd_names:
            continue
        type_names = set(audit.types)
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
    return xsd_diff


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #
def run_audit() -> Dict[str, object]:
    """Full human-oriented audit via live registry introspection."""
    _prepare_import_environment()

    entities: List[EntityAudit] = [
        _audit_entity(entity, module_path, attr)
        for entity, (module_path, attr) in sorted(ENTITY_FACTORIES.items())
    ]

    raven_files = _collect_py_files(REPO_ROOT / "ravenframework")
    plugins_root = REPO_ROOT / "plugins"
    plugin_files = _collect_py_files(plugins_root) if plugins_root.exists() else []
    manual_raven = _scan_manual_parsing(raven_files)
    manual_plugins = _scan_manual_parsing(plugin_files)

    xsd_diff = _compute_xsd_diff(entities)

    return {
        "entities": [audit.__dict__ for audit in entities],
        "manual_parsing": {
            "ravenframework": manual_raven,
            "plugins": manual_plugins,
        },
        "xsd_diff": xsd_diff,
    }


def coverage_facts(audit: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Minimal, portable regression baseline of coverage FACTS only.

    Per entity, keeps exactly the stable facts a guardrail should defend:
      * ``missing_spec``  -- registered types with no getInputSpecification
      * ``errors``        -- factory import / class resolve failures
      * ``extra_in_xsd``  -- types present in the XSD but no longer registered

    Deliberately excludes:
      * the volatile ``types`` rosters and the manual-parsing path map, and
      * ``missing_in_xsd`` -- a *newly added* well-formed type is, by nature,
        transiently absent from the hand-written XSD until it is regenerated,
        so freezing ``missing_in_xsd`` would make benign additions fail the
        test. ``missing_in_xsd`` is still reported by ``run_audit`` for humans;
        it is just not a frozen regression fact. (``extra_in_xsd`` is the
        non-churning direction: it only grows when a registered type is removed
        or the schema references a phantom type -- a real drift.)

    The net effect:
      * adding a well-formed entity (new type WITH a spec) -> ``missing_spec``
        empty and ``extra_in_xsd`` unchanged -> the baseline does NOT change;
      * removing a spec / introducing a manual-parse or unresolvable class ->
        ``missing_spec`` (or ``errors``) grows -> the test FAILS;
      * deleting a registered type while leaving it in the XSD, or an entity
        that no longer imports -> ``extra_in_xsd`` / ``errors`` grows -> FAILS.

    The result is deterministic (sorted keys/lists) and checkout-independent.
    """
    if audit is None:
        audit = run_audit()
    entities = audit.get("entities", [])  # type: ignore[union-attr]
    xsd_diff = audit.get("xsd_diff", {})  # type: ignore[union-attr]

    coverage: Dict[str, Dict[str, List[str]]] = {}
    for ent in entities:
        name = ent["entity"]
        coverage[name] = {
            "missing_spec": sorted(ent.get("missing_spec", [])),
            "errors": sorted(ent.get("errors", [])),
        }
    norm_xsd = {
        entity: {"extra_in_xsd": sorted(diff.get("extra_in_xsd", []))}
        for entity, diff in xsd_diff.items()
    }
    return {
        "schema_version": 1,
        "coverage": coverage,
        "xsd_diff": {entity: norm_xsd[entity] for entity in sorted(norm_xsd)},
    }


# --------------------------------------------------------------------------- #
# Text report.
# --------------------------------------------------------------------------- #
def _print_report(data: Dict[str, object]) -> None:
    entities = data.get("entities", [])
    print("InputData Spec Coverage Audit")
    print("=" * 32)
    for ent in entities:
        entity = ent["entity"]
        missing = ent["missing_spec"]
        unresolved = ent["unresolved"]
        errors = ent.get("errors", [])
        if not missing and not unresolved and not errors:
            continue
        print(f"\n[{entity}]")
        if errors:
            print(f"  Errors: {len(errors)}")
            for name in errors[:50]:
                print(f"    - {name}")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", help="Write full audit JSON report to path")
    parser.add_argument(
        "--baseline-json",
        dest="baseline_path",
        help="Write the minimal coverage-facts regression baseline to path",
    )
    parser.add_argument("--no-print", action="store_true", help="Skip text report")
    args = parser.parse_args()

    data = run_audit()
    if not args.no_print:
        _print_report(data)
    if args.json_path:
        Path(args.json_path).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    if args.baseline_path:
        Path(args.baseline_path).write_text(
            json.dumps(coverage_facts(data), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
