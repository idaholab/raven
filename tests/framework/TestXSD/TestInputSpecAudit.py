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
"""
Regression guardrail for InputData spec coverage.

Rather than freezing the entire audit (which bakes in volatile type rosters and
checkout-specific paths and so breaks on any benign addition or source drift),
this test compares only the STABLE COVERAGE FACTS against a committed baseline:

  * per entity, the SET of registered classes missing getInputSpecification
  * per entity, factory-import / class-resolve errors
  * the xsd_diff missing/extra-in-XSD sets

Discovery is live-registry based (see developer_tools/audit_input_specs.py), so
algorithms registered via registerAlgorithm-style sub-registries are covered too.

Guard semantics:
  * Adding a well-formed entity (new type WITH a spec, present in the XSD) does
    NOT fail this test -- its facts are empty, so the baseline is unchanged.
  * Removing a getInputSpecification, breaking XSD coverage, introducing a
    manual-parse class, or an entity that no longer imports DOES fail it.

To refresh the baseline after an intentional change:
    python3 developer_tools/sync_input_specs.py
See developer_tools/README.md for details.
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import json
import os
import sys

raven_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(raven_dir)

from developer_tools import audit_input_specs


def _normalize(obj):
  """Order-insensitive canonical form for stable comparison."""
  if isinstance(obj, dict):
    return {key: _normalize(obj[key]) for key in sorted(obj.keys())}
  if isinstance(obj, list):
    return sorted((_normalize(item) for item in obj), key=lambda x: json.dumps(x, sort_keys=True))
  return obj


def _diff_keys(current, baseline, path=""):
  """Return a list of human-readable discrepancies between two nested dicts."""
  problems = []
  if isinstance(baseline, dict) and isinstance(current, dict):
    for key in sorted(set(baseline) | set(current)):
      sub = f"{path}.{key}" if path else key
      if key not in current:
        problems.append(f"  missing in current: {sub}")
      elif key not in baseline:
        problems.append(f"  unexpected in current: {sub}")
      else:
        problems.extend(_diff_keys(current[key], baseline[key], sub))
  elif current != baseline:
    problems.append(f"  {path}: baseline={baseline!r} current={current!r}")
  return problems


def _load_baseline(path):
  with open(path, "r", encoding="utf-8") as handle:
    return json.load(handle)


baseline_path = os.path.join(raven_dir, "developer_tools", "audit_input_specs.json")
if not os.path.isfile(baseline_path):
  print("FAILED: missing baseline JSON:", baseline_path)
  sys.exit(1)

current = audit_input_specs.coverage_facts()
baseline = _load_baseline(baseline_path)

current_norm = _normalize(current)
baseline_norm = _normalize(baseline)

if current_norm != baseline_norm:
  print("FAILED: InputData spec coverage facts differ from baseline.")
  for line in _diff_keys(current_norm, baseline_norm):
    print(line)
  print("Hint: if this change is intentional, run developer_tools/sync_input_specs.py")
  print("      to refresh the baseline and generated XSDs.")
  print("See developer_tools/README.md for details.")
  sys.exit(1)

print("passes 1 fails 0")
sys.exit(0)

"""
  <TestInfo>
    <name>framework.test_input_spec_audit</name>
    <author>codex</author>
    <created>2026-02-09</created>
    <classesTested>developer_tools.audit_input_specs</classesTested>
    <description>
      Regression guardrail: asserts the stable InputData spec coverage facts
      (per-entity missing getInputSpecification sets, factory errors, and
      xsd_diff sets) match the committed baseline. Tolerates adding a
      well-formed entity; catches a removed spec, broken XSD coverage, or a
      newly introduced manual-parse / unresolvable class.
    </description>
    <revisions>
      <revision author="codex" date="2026-02-09">Initial version.</revision>
      <revision author="Jimmy-INL" date="2026-10-05">
        Switch from full-audit freeze to live-registry discovery plus a minimal,
        portable coverage-facts baseline.
      </revision>
    </revisions>
  </TestInfo>
"""
