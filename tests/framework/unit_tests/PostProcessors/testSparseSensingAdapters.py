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
import xarray as xr

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
check("default reconstruction method is auto", pp.reconstructionMethod == "auto")
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

xmlUQ = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <uncertaintyMetrics>std</uncertaintyMetrics>
    <uncertaintyPrior>1.0,0.5</uncertaintyPrior>
    <uncertaintyNoise>0.2</uncertaintyNoise>
    <reconstructionErrorRange>1,2</reconstructionErrorRange>
  </Goal>
</PostProcessor>"""

ppUQ = SparseSensing()
ppUQ._handleInput(parse(xmlUQ))
check("uncertainty metrics are normalized", ppUQ.uncertaintyMetrics == ["std"])
check("uncertainty prior is parsed into a vector",
      np.allclose(ppUQ.uncertaintyPrior, np.asarray([1.0, 0.5])))
check("uncertainty noise is parsed", abs(ppUQ.uncertaintyNoise - 0.2) < 1e-14)
check("reconstructionErrorRange is parsed",
      np.array_equal(ppUQ.reconstructionErrorRange, np.asarray([1, 2])))

xmlReconstructionRegularized = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <reconstructionMethod>regularized</reconstructionMethod>
    <reconstructionPrior>1.0,0.25</reconstructionPrior>
    <reconstructionNoise>0.2</reconstructionNoise>
    <reconstructionRankTolerance>1e-8</reconstructionRankTolerance>
  </Goal>
</PostProcessor>"""

ppReconstructionRegularized = SparseSensing()
ppReconstructionRegularized._handleInput(parse(xmlReconstructionRegularized))
check("regularized reconstruction method is parsed",
      ppReconstructionRegularized.reconstructionMethod == "regularized")
check("reconstruction prior is parsed into a vector",
      np.allclose(ppReconstructionRegularized.reconstructionPrior, np.asarray([1.0, 0.25])))
check("reconstruction noise is parsed",
      abs(ppReconstructionRegularized.reconstructionNoise - 0.2) < 1e-14)
check("reconstruction rank tolerance is parsed",
      abs(ppReconstructionRegularized.reconstructionRankTolerance - 1e-8) < 1e-14)

xmlTPGREnergy = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>TPGR</optimizer>
    <energyLandscapeMetrics>one_pt,two_pt</energyLandscapeMetrics>
    <energyLandscapeSensors>1,3</energyLandscapeSensors>
    <uncertaintyNoise>0.2</uncertaintyNoise>
  </Goal>
</PostProcessor>"""

ppTPGREnergy = SparseSensing()
ppTPGREnergy._handleInput(parse(xmlTPGREnergy))
check("energy-landscape metrics are normalized", ppTPGREnergy.energyLandscapeMetrics == ["one_pt", "two_pt"])
check("energy-landscape sensor list is parsed",
      np.array_equal(ppTPGREnergy.energyLandscapeSensors, np.asarray([1, 3])))

xmlGQRExact = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>GQR</optimizer>
    <constraint strategy='exact_n'>
      <indices>4,8</indices>
      <nConstSensors>1</nConstSensors>
    </constraint>
  </Goal>
</PostProcessor>"""

ppGQRExact = SparseSensing()
ppGQRExact._handleInput(parse(xmlGQRExact))
check("GQR exact_n strategy is parsed", ppGQRExact.constraintSpec["strategy"] == "exact_n")
check("GQR exact_n indices are parsed", ppGQRExact.constraintSpec["indices"] == [4, 8])

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

xmlClassification = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='classification'>
    <features>x,T</features>
    <target>T</target>
    <label>classLabel</label>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <l1Penalty>0.2</l1Penalty>
    <threshold>1e-10</threshold>
  </Goal>
</PostProcessor>"""

ppClass = SparseSensing()
ppClass._handleInput(parse(xmlClassification))
check("classification goal is parsed", ppClass.sparseSensingGoal == "classification")
check("classification label is parsed", ppClass.classificationLabel == "classLabel")
check("classification l1 penalty is parsed", abs(ppClass.classificationL1Penalty - 0.2) < 1e-14)
check("classification threshold is parsed", abs(ppClass.classificationThreshold - 1e-10) < 1e-20)
check("classification builder returns SSPOC",
      ppClass._buildModel(ppClass._buildBasis(), None).__class__.__name__ == "SSPOC")

checkRaises("classification requires a label node", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='classification'>
    <features>x,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
  </Goal>
</PostProcessor>""")), "label")

checkRaises("negative classification l1Penalty is rejected", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='classification'>
    <features>x,T</features>
    <target>T</target>
    <label>classLabel</label>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <l1Penalty>-0.1</l1Penalty>
  </Goal>
</PostProcessor>""")), "non-negative")

xmlClassificationRun = """<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='classification'>
    <features>x,T</features>
    <target>T</target>
    <label>classLabel</label>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
  </Goal>
</PostProcessor>"""
ppClassRun = SparseSensing()
ppClassRun._handleInput(parse(xmlClassificationRun))
field = np.asarray([[0.0, 0.0, 1.0, 1.0],
                    [0.1, 0.0, 1.0, 0.9],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.9, 1.0, 0.1, 0.0]])
sampleCoord = np.arange(field.shape[0])
sensorCoord = np.arange(field.shape[1])
inputDS = xr.Dataset({
  "x": (("RAVEN_sample_ID", "index"), np.tile(sensorCoord, (field.shape[0], 1))),
  "T": (("RAVEN_sample_ID", "index"), field),
  "classLabel": ("RAVEN_sample_ID", np.asarray([0, 0, 1, 1])),
}, coords={"RAVEN_sample_ID": sampleCoord, "index": sensorCoord})
classOut = ppClassRun.run({"Data": [(None, None, inputDS)]})
check("classification run outputs one row per selected sensor",
      classOut.sizes["sensor"] == ppClassRun.nSensors)
check("classification run outputs selected sensor coordinates",
      "x" in classOut and classOut["x"].sizes["sensor"] == ppClassRun.nSensors)
check("classification run outputs measured target values at sensors",
      "T" in classOut and classOut["T"].sizes["sensor"] == ppClassRun.nSensors)

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

checkRaises("unknown uncertainty metric is rejected", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <uncertaintyMetrics>bogus</uncertaintyMetrics>
  </Goal>
</PostProcessor>""")), "uncertainty metric")

checkRaises("unknown energyLandscape metric is rejected", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>TPGR</optimizer>
    <energyLandscapeMetrics>bogus</energyLandscapeMetrics>
  </Goal>
</PostProcessor>""")), "energyLandscape metric")

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

checkRaises("uncertaintyPrior must be numeric when not decreasing", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <uncertaintyMetrics>std</uncertaintyMetrics>
    <uncertaintyPrior>oops</uncertaintyPrior>
  </Goal>
</PostProcessor>""")), "uncertaintyPrior")

checkRaises("reconstructionPrior must be numeric when not decreasing", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <reconstructionMethod>regularized</reconstructionMethod>
    <reconstructionPrior>oops</reconstructionPrior>
  </Goal>
</PostProcessor>""")), "reconstructionPrior")

checkRaises("negative reconstructionNoise is rejected", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <reconstructionNoise>-0.1</reconstructionNoise>
  </Goal>
</PostProcessor>""")), "non-negative")

checkRaises("nonpositive reconstructionRankTolerance is rejected", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <reconstructionRankTolerance>0</reconstructionRankTolerance>
  </Goal>
</PostProcessor>""")), "positive")

checkRaises("reconstructionErrorRange requires positive integers", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>QR</optimizer>
    <reconstructionErrorRange>0,2</reconstructionErrorRange>
  </Goal>
</PostProcessor>""")), "positive integers")

checkRaises("energyLandscapeSensors requires non-negative integers", IOError,
            lambda: SparseSensing()._handleInput(parse("""<PostProcessor name='pp' subType='SparseSensing'>
  <Goal subType='reconstruction'>
    <features>X,Y,T</features>
    <target>T</target>
    <basis>SVD</basis>
    <nModes>2</nModes>
    <nSensors>2</nSensors>
    <optimizer>TPGR</optimizer>
    <energyLandscapeMetrics>two_pt</energyLandscapeMetrics>
    <energyLandscapeSensors>-1</energyLandscapeSensors>
  </Goal>
</PostProcessor>""")), "non-negative")

ppMetric = SparseSensing()
ppMetric.sparseSensingGoal = "reconstruction"
data = np.asarray([[1.0, 2.0], [1.5, 2.5], [2.0, 3.0]])
ppMetric.nModes = 1
ppMetric.nSensors = 1
ppMetric.basis = "Identity"
ppMetric.optimizer = "QR"
model = ppMetric._buildModel(ppMetric._buildBasis(), ppMetric._buildOptimizer())
model.fit(data)
# Build the unified reconstruction-metric registry (assembler + shorthand desugar) and route
# evaluation through the same code path users hit at run time.
ppMetric.reconstructionMetrics = ["rmse", "mse", "mae"]
ppMetric.assemblerDict = {}
ppMetric._buildMetricsDict()
rmse = ppMetric._evaluateRavenMetric(model, data, ppMetric.metricsDict["rec_rmse"])
mse = ppMetric._evaluateRavenMetric(model, data, ppMetric.metricsDict["rec_mse"])
mae = ppMetric._evaluateRavenMetric(model, data, ppMetric.metricsDict["rec_mae"])
check("rmse metric is non-negative", rmse >= 0.0)
check("mse metric is non-negative", mse >= 0.0)
check("mae metric is non-negative", mae >= 0.0)
check("rmse and mse remain numerically consistent", abs(rmse**2 - mse) < 1e-12)

ppAuto = SparseSensing()
ppAuto.sparseSensingGoal = "reconstruction"
ppAuto.nModes = 1
ppAuto.nSensors = 1
ppAuto.basis = "SVD"
ppAuto.optimizer = "QR"
rankOneData = np.asarray([[1.0, 2.0, 3.0],
                          [2.0, 4.0, 6.0],
                          [3.0, 6.0, 9.0]])
autoModel = ppAuto._buildModel(ppAuto._buildBasis(), ppAuto._buildOptimizer())
autoModel.fit(rankOneData)
check("auto reconstruction method uses unregularized for basis-consistent data",
      ppAuto._resolveReconstructionMethod(autoModel, rankOneData) == "unregularized")

ppAutoRegularized = SparseSensing()
ppAutoRegularized.sparseSensingGoal = "reconstruction"
ppAutoRegularized.nModes = 1
ppAutoRegularized.nSensors = 1
ppAutoRegularized.basis = "SVD"
ppAutoRegularized.optimizer = "QR"
fullRankData = np.eye(3)
autoRegularizedModel = ppAutoRegularized._buildModel(ppAutoRegularized._buildBasis(), ppAutoRegularized._buildOptimizer())
autoRegularizedModel.fit(fullRankData)
check("auto reconstruction method uses regularized when retained basis has residual",
      ppAutoRegularized._resolveReconstructionMethod(autoRegularizedModel, fullRankData) == "regularized")
ppAutoRegularized.reconstructionNoise = 0.0
check("auto reconstruction method treats explicit zero noise as unregularized",
      ppAutoRegularized._resolveReconstructionMethod(autoRegularizedModel, fullRankData) == "unregularized")

ppExplicitRegularized = SparseSensing()
ppExplicitRegularized.sparseSensingGoal = "reconstruction"
ppExplicitRegularized.nModes = 1
ppExplicitRegularized.nSensors = 1
ppExplicitRegularized.basis = "SVD"
ppExplicitRegularized.optimizer = "QR"
ppExplicitRegularized.reconstructionMethod = "regularized"
ppExplicitRegularized.reconstructionNoise = 0.2
explicitRegularizedModel = ppExplicitRegularized._buildModel(ppExplicitRegularized._buildBasis(), ppExplicitRegularized._buildOptimizer())
explicitRegularizedModel.fit(fullRankData)
regularizedPrediction = ppExplicitRegularized._predictFullState(explicitRegularizedModel, fullRankData)
check("explicit regularized reconstruction returns the full-state shape",
      regularizedPrediction.shape == fullRankData.shape)

ppStd = SparseSensing()
ppStd.sparseSensingGoal = "reconstruction"
ppStd.nModes = 2
ppStd.nSensors = 2
ppStd.basis = "SVD"
ppStd.optimizer = "QR"
ppStd.uncertaintyMetrics = ["std"]
ppStd.uncertaintyPrior = "decreasing"
stdData = np.asarray([[1.0, 2.0, 3.0, 4.0],
                      [1.2, 2.1, 2.8, 4.1],
                      [0.8, 1.9, 3.2, 3.9]])
stdModel = ppStd._buildModel(ppStd._buildBasis(), ppStd._buildOptimizer())
stdModel.fit(stdData)
stdValues = ppStd._computeUncertaintyMetric(stdModel, "std")
check("std uncertainty metric matches feature count", stdValues.shape == (4,))
check("std uncertainty metric is non-negative", np.all(stdValues >= 0.0))

ppStd.uncertaintyPrior = np.asarray([1.0])
checkRaises("uncertaintyPrior length must match retained modes", IOError,
            lambda: ppStd._resolveUncertaintyPrior(stdModel), "expected 2 values")

ppCurve = SparseSensing()
ppCurve.sparseSensingGoal = "reconstruction"
ppCurve.nModes = 2
ppCurve.nSensors = 2
ppCurve.basis = "SVD"
ppCurve.optimizer = "QR"
ppCurve.reconstructionErrorRange = np.asarray([1, 2])
curveModel = ppCurve._buildModel(ppCurve._buildBasis(), ppCurve._buildOptimizer())
curveModel.fit(stdData)
curve = curveModel.reconstruction_error(stdData, sensor_range=ppCurve.reconstructionErrorRange)
check("reconstruction_error returns one value per requested sensor count",
      np.asarray(curve).shape == (2,))
check("reconstruction_error values are non-negative", np.all(np.asarray(curve) >= 0.0))

ppTPGRCurve = SparseSensing()
ppTPGRCurve.sparseSensingGoal = "reconstruction"
ppTPGRCurve.nModes = 2
ppTPGRCurve.nSensors = 2
ppTPGRCurve.basis = "SVD"
ppTPGRCurve.optimizer = "TPGR"
ppTPGRCurve.reconstructionErrorRange = np.asarray([1, 2])
tpgrCurveModel = ppTPGRCurve._buildModel(ppTPGRCurve._buildBasis(), ppTPGRCurve._buildOptimizer())
tpgrCurveModel.fit(stdData, seed=0)
checkRaises("TPGR reconstruction_error output is rejected explicitly", IOError,
            lambda: ppTPGRCurve._addReconstructionErrorOutputs(None, tpgrCurveModel, stdData),
            'not supported with optimizer "TPGR"')

ppTPGREnergyRuntime = SparseSensing()
ppTPGREnergyRuntime.sparseSensingGoal = "reconstruction"
ppTPGREnergyRuntime.nModes = 2
ppTPGREnergyRuntime.nSensors = 2
ppTPGREnergyRuntime.basis = "SVD"
ppTPGREnergyRuntime.optimizer = "TPGR"
ppTPGREnergyRuntime.energyLandscapeMetrics = ["one_pt", "two_pt"]
ppTPGREnergyRuntime.energyLandscapeSensors = np.asarray([1, 3])
ppTPGREnergyRuntime.uncertaintyPrior = "decreasing"
ppTPGREnergyRuntime.uncertaintyNoise = 0.2
tpgrEnergyModel = ppTPGREnergyRuntime._buildModel(ppTPGREnergyRuntime._buildBasis(), ppTPGREnergyRuntime._buildOptimizer())
tpgrEnergyModel.fit(stdData, seed=0)
onePt = ppTPGREnergyRuntime._computeEnergyLandscapeMetric(tpgrEnergyModel, "one_pt")
twoPt = ppTPGREnergyRuntime._computeEnergyLandscapeMetric(tpgrEnergyModel, "two_pt")
check("TPGR one-point energy landscape matches feature count", onePt.shape == (4,))
check("TPGR one-point energy landscape is finite", np.all(np.isfinite(onePt)))
check("TPGR two-point energy landscape matches feature count", twoPt.shape == (4,))
check("TPGR two-point energy landscape masks selected sensors with NaN",
      np.isnan(twoPt[1]) and np.isnan(twoPt[3]))

ppMissingTwoPt = SparseSensing()
ppMissingTwoPt.sparseSensingGoal = "reconstruction"
ppMissingTwoPt.nModes = 2
ppMissingTwoPt.nSensors = 2
ppMissingTwoPt.basis = "SVD"
ppMissingTwoPt.optimizer = "TPGR"
ppMissingTwoPt.energyLandscapeMetrics = ["two_pt"]
ppMissingTwoPt.uncertaintyPrior = "decreasing"
ppMissingTwoPt.uncertaintyNoise = 0.2
checkRaises("two-point energy landscape requires selected sensors", IOError,
            lambda: ppMissingTwoPt._computeEnergyLandscapeMetric(tpgrEnergyModel, "two_pt"),
            "energyLandscapeSensors is required")

ppBadEnergyOptimizer = SparseSensing()
ppBadEnergyOptimizer.sparseSensingGoal = "reconstruction"
ppBadEnergyOptimizer.nModes = 2
ppBadEnergyOptimizer.nSensors = 2
ppBadEnergyOptimizer.basis = "SVD"
ppBadEnergyOptimizer.optimizer = "QR"
ppBadEnergyOptimizer.energyLandscapeMetrics = ["one_pt"]
ppBadEnergyOptimizer.uncertaintyPrior = "decreasing"
ppBadEnergyOptimizer.uncertaintyNoise = 0.2
checkRaises("energy landscapes are rejected for non-TPGR optimizers", IOError,
            lambda: ppBadEnergyOptimizer._computeEnergyLandscapeMetric(curveModel, "one_pt"),
            'optimizer "TPGR"')

print("Results:", results)
sys.exit(results["fail"])
