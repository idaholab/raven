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
  Unit tests for SparseSensing factory helpers on the devel-based alignment branch.
"""
import os
import sys
import xml.etree.ElementTree as ET

ravenDir = os.path.abspath(os.path.join(*([os.path.dirname(__file__)]+[os.pardir]*4)))
sys.path.append(ravenDir)

from ravenframework.Models.PostProcessors.SparseSensing import SparseSensing

results = {"pass": 0, "fail": 0}

def check(comment, condition):
  """Record a boolean check result."""
  if condition:
    results["pass"] += 1
  else:
    print("check failed:", comment)
    results["fail"] += 1

def checkRaises(comment, excType, func, contains=None):
  """Record whether func raises the expected exception."""
  try:
    func()
  except excType as err:
    if contains is not None and contains not in str(err):
      print("check failed:", comment, "missing text:", contains, "actual:", err)
      results["fail"] += 1
      return
    results["pass"] += 1
    return
  except Exception as err:
    print("check failed:", comment, "wrong exception:", type(err).__name__, err)
    results["fail"] += 1
    return
  print("check failed:", comment, "no exception raised")
  results["fail"] += 1

def parse(xmlString):
  """Parse a SparseSensing XML fragment into a ParameterInput."""
  spec = SparseSensing.getInputSpecification()()
  spec.parseNode(ET.fromstring(xmlString))
  return spec

xmlCanonical = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>RandomProjection</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
  </Goal>
</PostProcessor>"""

xmlLegacy = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>RandomProjetion</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
  </Goal>
</PostProcessor>"""

pp = SparseSensing()
pp._handleInput(parse(xmlCanonical))
check("canonical RandomProjection spelling is preserved", pp.basis == "RandomProjection")
check("RandomProjection builder returns the right class",
      pp._buildBasis().__class__.__name__ == "RandomProjection")
check("QR builder returns the right class",
      pp._buildOptimizer().__class__.__name__ == "QR")
check("reconstruction builder returns SSPOR",
      pp._buildModel(pp._buildBasis(), pp._buildOptimizer()).__class__.__name__ == "SSPOR")

ppLegacy = SparseSensing()
ppLegacy._handleInput(parse(xmlLegacy))
check("legacy RandomProjetion spelling is normalized", ppLegacy.basis == "RandomProjection")
check("legacy spelling still builds RandomProjection",
      ppLegacy._buildBasis().__class__.__name__ == "RandomProjection")

ppBadBasis = SparseSensing()
ppBadBasis.basis = "Bogus"
ppBadBasis.nModes = 2
checkRaises("unknown basis is rejected", IOError, ppBadBasis._buildBasis, "not recognized")

ppClass = SparseSensing()
ppClass.sparseSensingGoal = "classification"
ppClass.nSensors = 2
checkRaises("classification is rejected explicitly", NotImplementedError,
            lambda: ppClass._buildModel(None, None), "not yet implemented")

print("Results:", results)
sys.exit(results["fail"])
