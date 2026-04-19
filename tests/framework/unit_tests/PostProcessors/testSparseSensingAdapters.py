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
import numpy as np

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

pp = SparseSensing()
pp._handleInput(parse(xmlCanonical))
check("canonical RandomProjection spelling is preserved", pp.basis == "RandomProjection")
check("RandomProjection builder returns the right class",
      pp._buildBasis().__class__.__name__ == "RandomProjection")
check("QR builder returns the right class",
      pp._buildOptimizer().__class__.__name__ == "QR")
check("reconstruction builder returns SSPOR",
      pp._buildModel(pp._buildBasis(), pp._buildOptimizer()).__class__.__name__ == "SSPOR")

xmlCCQR = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>CCQR</optimizer>
    <sensorCosts>sensorCost</sensorCosts>
    <reconstructionMetrics>RMSE,mae</reconstructionMetrics>
  </Goal>
</PostProcessor>"""

ppCCQR = SparseSensing()
ppCCQR._handleInput(parse(xmlCCQR))
ppCCQR.sensorCosts = [0.25, 0.75, 1.25]
check("CCQR optimizer name is parsed", ppCCQR.optimizer == "CCQR")
check("sensorCosts variable name is parsed", ppCCQR.sensorCostsVariableName == "sensorCost")
check("reconstruction metrics are normalized", ppCCQR.reconstructionMetrics == ["rmse", "mae"])
check("CCQR builder returns the right class",
      ppCCQR._buildOptimizer().__class__.__name__ == "CCQR")

xmlGQR = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>GQR</optimizer>
    <constraint strategy='max_n'>
      <shape>Circle</shape>
      <xAxis>X</xAxis>
      <yAxis>Y</yAxis>
      <centerX>1.0</centerX>
      <centerY>1.0</centerY>
      <radius>0.25</radius>
      <loc>in</loc>
      <nConstSensors>0</nConstSensors>
    </constraint>
  </Goal>
</PostProcessor>"""

ppGQR = SparseSensing()
ppGQR._handleInput(parse(xmlGQR))
check("GQR optimizer name is parsed", ppGQR.optimizer == "GQR")
check("GQR builder returns the right class",
      ppGQR._buildOptimizer().__class__.__name__ == "GQR")
check("GQR constraint strategy is parsed", ppGQR.constraintSpec["strategy"] == "max_n")
check("GQR constraint shape is parsed", ppGQR.constraintSpec["shape"] == "Circle")
check("GQR constraint xAxis is parsed", ppGQR.constraintSpec["xAxis"] == "X")

xmlTPGR = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>TPGR</optimizer>
  </Goal>
</PostProcessor>"""

ppTPGR = SparseSensing()
ppTPGR._handleInput(parse(xmlTPGR))
check("TPGR optimizer name is parsed", ppTPGR.optimizer == "TPGR")
check("TPGR builder returns the right class",
      ppTPGR._buildOptimizer().__class__.__name__ == "TPGR")

checkRaises("legacy RandomProjetion spelling is rejected", IOError,
            lambda: parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>RandomProjetion</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
  </Goal>
</PostProcessor>"""))

ppBadBasis = SparseSensing()
ppBadBasis.basis = "Bogus"
ppBadBasis.nModes = 2
checkRaises("unknown basis is rejected", IOError, ppBadBasis._buildBasis, "not recognized")

ppClass = SparseSensing()
ppClass.sparseSensingGoal = "classification"
ppClass.nSensors = 2
checkRaises("classification is rejected explicitly", NotImplementedError,
            lambda: ppClass._buildModel(None, None), "not yet implemented")

checkRaises("unknown reconstruction metric is rejected", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <reconstructionMetrics>rmse,bogus</reconstructionMetrics>
  </Goal>
</PostProcessor>""")), "reconstruction metric")

checkRaises("GQR max_n requires nConstSensors", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>GQR</optimizer>
    <constraint strategy='max_n'>
      <shape>Circle</shape>
      <xAxis>X</xAxis>
      <yAxis>Y</yAxis>
      <centerX>1.0</centerX>
      <centerY>1.0</centerY>
      <radius>0.25</radius>
    </constraint>
  </Goal>
</PostProcessor>""")), "nConstSensors")

checkRaises("GQR distance requires radius", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>GQR</optimizer>
    <constraint strategy='distance'>
      <xAxis>X</xAxis>
      <yAxis>Y</yAxis>
    </constraint>
  </Goal>
</PostProcessor>""")), "radius")

ppMetric = SparseSensing()
ppMetric.sparseSensingGoal = "reconstruction"
data = np.asarray([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
ppMetric.nModes = 1
ppMetric.nSensors = 1
ppMetric.basis = "Identity"
ppMetric.optimizer = "QR"
model = ppMetric._buildModel(ppMetric._buildBasis(), ppMetric._buildOptimizer())
model.fit(data)
rmse = ppMetric._computeReconstructionMetric(model, data, "rmse")
mse = ppMetric._computeReconstructionMetric(model, data, "mse")
mae = ppMetric._computeReconstructionMetric(model, data, "mae")
check("rmse metric is non-negative", rmse >= 0.0)
check("mse metric is non-negative", mse >= 0.0)
check("mae metric is non-negative", mae >= 0.0)
check("rmse and mse remain numerically consistent", abs(rmse**2 - mse) < 1e-12)

print("Results:", results)
sys.exit(results["fail"])
