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
Compare InputData audit output against the stored baseline JSON.
"""
from __future__ import division, print_function, unicode_literals, absolute_import

import json
import os
import sys

raven_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(raven_dir)

from developer_tools import audit_input_specs


def _normalize(obj):
  if isinstance(obj, dict):
    return {key: _normalize(obj[key]) for key in sorted(obj.keys())}
  if isinstance(obj, list):
    return sorted((_normalize(item) for item in obj), key=lambda x: json.dumps(x, sort_keys=True))
  return obj


def _load_baseline(path):
  with open(path, "r", encoding="utf-8") as handle:
    return json.load(handle)


baseline_path = os.path.join(raven_dir, "developer_tools", "audit_input_specs.json")
if not os.path.isfile(baseline_path):
  print("FAILED: missing baseline JSON:", baseline_path)
  sys.exit(1)

current = audit_input_specs.run_audit()
baseline = _load_baseline(baseline_path)

current_norm = _normalize(current)
baseline_norm = _normalize(baseline)

if current_norm != baseline_norm:
  print("FAILED: InputData audit differs from baseline JSON.")
  print("Hint: run developer_tools/sync_input_specs.py to refresh the baseline and generated XSDs.")
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
      Ensures InputData audit output matches the stored baseline JSON.
    </description>
    <revisions>
      <revision author="codex" date="2026-02-09">Initial version.</revision>
    </revisions>
  </TestInfo>
"""
