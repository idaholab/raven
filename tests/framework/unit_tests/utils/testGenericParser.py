# Copyright 2017 Battelle Energy Alliance, LLC
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
  Unit tests for GenericParser scalar formatting.
"""
import os
import sys
import numpy as np

ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(ravenDir)

from ravenframework.CodeInterfaceClasses.Generic.GenericParser import _reprIfFloat

results = {"pass": 0, "fail": 0}

def check(comment, value, expected):
  """Compare a serialized value against its expected string."""
  if value == expected:
    results["pass"] += 1
  else:
    print("checking answer", comment, value, "!=", expected)
    results["fail"] += 1

check("python float uses repr", _reprIfFloat(1.25), repr(1.25))
check("numpy float scalar drops constructor wrapper", _reprIfFloat(np.float64(1.25)), repr(1.25))
check("numpy float32 scalar drops constructor wrapper", _reprIfFloat(np.float32(2.5)), repr(float(np.float32(2.5))))
check("numpy float scalar keeps round-trip precision",
      _reprIfFloat(np.float64(994.49329926271298)), "994.49329926271298")
check("numpy integer scalar uses plain integer text", _reprIfFloat(np.int64(7)), "7")
check("python integer uses plain integer text", _reprIfFloat(3), "3")
check("strings pass through unchanged", _reprIfFloat("power"), "power")

print(results)
sys.exit(results["fail"])
