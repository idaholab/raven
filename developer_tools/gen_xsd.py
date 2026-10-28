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
"""
Generate a full RAVEN XSD based on InputData.
"""
import os
import sys
import builtins
import xml.etree.ElementTree as ET

try:
  import ravenframework
except ModuleNotFoundError:
  sys.path.append(os.path.dirname(os.path.dirname(__file__)))
  import ravenframework

builtins.profile = lambda f: f
from ravenframework.utils import InputData
import ravenframework.Simulation

if len(sys.argv) != 2:
  print(sys.argv[0], "generated_filename.xsd")
  sys.exit(1)

if os.environ.get("RAVEN_SUPPRESS_INPUT_SPEC_WARNINGS", "").lower() in ("1", "true", "yes"):
  InputData.SUPPRESS_INPUT_SPEC_WARNINGS = True

base = ravenframework.Simulation.Simulation.getXSDSchema()
ET.ElementTree(base).write(sys.argv[1])
print("Generated", sys.argv[1])
