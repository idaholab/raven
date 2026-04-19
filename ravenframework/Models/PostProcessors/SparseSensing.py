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
'''
  Created on May 24, 2022
  @ Authors: Mohammad Abdo (@Jimmy-INL)
             Niharika Karnik (@nkarnik)
'''
import pysensors as ps
import numpy as np
import pandas as pd
import xarray as xr

from .PostProcessorReadyInterface import PostProcessorReadyInterface
from ...utils import InputData, InputTypes

class SparseSensing(PostProcessorReadyInterface):
  """
    This Postprocessor class finds the optimal locations of sparse sensors for both classification and reconstruction problems.
    The implemention utilizes the opensource library pysensors and is based on following publications:
    - Brunton, Bingni W., et al. "Sparse sensor placement optimization for classification." SIAM Journal on Applied Mathematics 76.5 (2016): 2099-2122.
    - Manohar, Krithika, et al. "Data-driven sparse sensor placement for reconstruction: Demonstrating the benefits of exploiting known patterns." IEEE Control Systems Magazine 38.3 (2018): 63-86.
    - de Silva, Brian M., et al. "PySensors: A Python package for sparse sensor placement." arXiv preprint arXiv:2102.13476 (2021).
  """
  goalsDict = {'reconstruction':r"""Sparse sensor placement Optimization for Reconstruction (SSPOR)""",
          'classification':r"""Sparse sensor placement Optimization for Classification (SSPOC)"""}
  basisOptions = ['Identity', 'SVD', 'RandomProjection']
  basisAliases = {'identity': 'Identity',
                  'svd': 'SVD',
                  'randomprojection': 'RandomProjection'}
  optimizerOptions = ['QR', 'CCQR', 'GQR', 'TPGR']
  optimizerAliases = {'qr': 'QR',
                      'ccqr': 'CCQR',
                      'gqr': 'GQR',
                      'tpgr': 'TPGR'}
  reconstructionMetricOptions = ['rmse', 'mse', 'mae']
  constraintStrategyOptions = ['max_n', 'exact_n', 'predetermined', 'distance']
  constraintShapeOptions = ['Circle', 'Cylinder', 'Line', 'Parabola', 'Ellipse', 'Polygon', 'UserDefined']
  constraintLocationOptions = ['in', 'out']

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    inputSpecification = super(SparseSensing, cls).getInputSpecification()
    goal = InputData.parameterInputFactory('Goal',
                                                  printPriority=108,
                                                  descr=r"""The goal of the sparse sensor optimization (i.e., reconstruction or classification)""")
    goal.addParam("subType", InputTypes.makeEnumType("Goal", "GoalType", ['reconstruction','classification']), False, default='reconstruction')
    features = InputData.parameterInputFactory("features", contentType=InputTypes.StringListType,
                                                printPriority=108,
                                                descr=r"""Features/inputs of the data model""")
    goal.addSub(features)
    target = InputData.parameterInputFactory("target", contentType=InputTypes.StringType,
                                                printPriority=108,
                                                descr=r"""target of data model""")
    goal.addSub(target)
    basis = InputData.parameterInputFactory("basis", contentType=InputTypes.makeEnumType("basis","basis Type", cls.basisOptions),
                                                           printPriority=108,
                                                           descr=r"""The type of basis onto which the data are projected""", default='SVD')
    goal.addSub(basis)
    nModes = InputData.parameterInputFactory("nModes", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The number of modes retained""")
    goal.addSub(nModes)
    nSensors = InputData.parameterInputFactory("nSensors", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The number of sensors used""")
    goal.addSub(nSensors)
    optimizer = InputData.parameterInputFactory("optimizer", contentType=InputTypes.makeEnumType("optimizer","optimizer type", cls.optimizerOptions),
                                                           printPriority=108,
                                                           descr=r"""The type of optimizer used""",default='QR')
    goal.addSub(optimizer)
    sensorCosts = InputData.parameterInputFactory("sensorCosts", contentType=InputTypes.StringType,
                                                  printPriority=108,
                                                  descr=r"""Name of the variable in the input DataObject that stores
                                                        the per-sensor costs used by the CCQR optimizer.""")
    goal.addSub(sensorCosts)
    constraint = InputData.parameterInputFactory("constraint",
                                                 printPriority=108,
                                                 descr=r"""Optional constraint definition used by the \xmlString{GQR}
                                                       optimizer.""")
    constraint.addParam("strategy",
                        InputTypes.makeEnumType("constraintStrategy", "constraint strategy type",
                                                cls.constraintStrategyOptions),
                        True)
    shape = InputData.parameterInputFactory("shape",
                                            contentType=InputTypes.makeEnumType("constraintShape", "constraint shape type",
                                                                                cls.constraintShapeOptions),
                                            printPriority=108,
                                            descr=r"""Built-in constrained region shape used to compute constrained sensor
                                                  indices for the \xmlString{GQR} optimizer.""")
    constraint.addSub(shape)
    indices = InputData.parameterInputFactory("indices", contentType=InputTypes.IntegerListType,
                                              printPriority=108,
                                              descr=r"""Explicit list of constrained sensor indices for \xmlString{GQR}.""")
    constraint.addSub(indices)
    nConstSensors = InputData.parameterInputFactory("nConstSensors", contentType=InputTypes.IntegerType,
                                                    printPriority=108,
                                                    descr=r"""Number of sensors allowed or prescribed in the constrained
                                                          region, depending on the chosen \xmlAttr{strategy}.""")
    constraint.addSub(nConstSensors)
    xAxis = InputData.parameterInputFactory("xAxis", contentType=InputTypes.StringType,
                                            printPriority=108,
                                            descr=r"""Feature variable used as the x-coordinate for dataframe-based
                                                  constrained sensing.""")
    constraint.addSub(xAxis)
    yAxis = InputData.parameterInputFactory("yAxis", contentType=InputTypes.StringType,
                                            printPriority=108,
                                            descr=r"""Feature variable used as the y-coordinate for dataframe-based
                                                  constrained sensing.""")
    constraint.addSub(yAxis)
    zAxis = InputData.parameterInputFactory("zAxis", contentType=InputTypes.StringType,
                                            printPriority=108,
                                            descr=r"""Feature variable used as the z-coordinate for 3-D constrained
                                                  sensing shapes such as \xmlString{Cylinder}.""")
    constraint.addSub(zAxis)
    loc = InputData.parameterInputFactory("loc",
                                          contentType=InputTypes.makeEnumType("constraintLocation", "constraint location type",
                                                                              cls.constraintLocationOptions),
                                          printPriority=108,
                                          descr=r"""Whether the inside or outside of the shape is constrained.
                                                Used by shape-based \xmlString{GQR} constraints.""")
    constraint.addSub(loc)
    centerX = InputData.parameterInputFactory("centerX", contentType=InputTypes.FloatType, printPriority=108)
    centerY = InputData.parameterInputFactory("centerY", contentType=InputTypes.FloatType, printPriority=108)
    centerZ = InputData.parameterInputFactory("centerZ", contentType=InputTypes.FloatType, printPriority=108)
    radius = InputData.parameterInputFactory("radius", contentType=InputTypes.FloatType, printPriority=108)
    height = InputData.parameterInputFactory("height", contentType=InputTypes.FloatType, printPriority=108)
    width = InputData.parameterInputFactory("width", contentType=InputTypes.FloatType, printPriority=108)
    angle = InputData.parameterInputFactory("angle", contentType=InputTypes.FloatType, printPriority=108)
    x1 = InputData.parameterInputFactory("x1", contentType=InputTypes.FloatType, printPriority=108)
    x2 = InputData.parameterInputFactory("x2", contentType=InputTypes.FloatType, printPriority=108)
    y1 = InputData.parameterInputFactory("y1", contentType=InputTypes.FloatType, printPriority=108)
    y2 = InputData.parameterInputFactory("y2", contentType=InputTypes.FloatType, printPriority=108)
    parabolaH = InputData.parameterInputFactory("h", contentType=InputTypes.FloatType, printPriority=108)
    parabolaK = InputData.parameterInputFactory("k", contentType=InputTypes.FloatType, printPriority=108)
    parabolaA = InputData.parameterInputFactory("a", contentType=InputTypes.FloatType, printPriority=108)
    equation = InputData.parameterInputFactory("equation", contentType=InputTypes.StringType, printPriority=108)
    fileNode = InputData.parameterInputFactory("file", contentType=InputTypes.StringType, printPriority=108)
    vertex = InputData.parameterInputFactory("vertex", contentType=InputTypes.FloatListType, printPriority=108,
                                             descr=r"""A polygon vertex encoded as \xmlString{x,y}.""")
    for sub in [centerX, centerY, centerZ, radius, height, width, angle,
                x1, x2, y1, y2, parabolaH, parabolaK, parabolaA,
                equation, fileNode, vertex]:
      constraint.addSub(sub)
    goal.addSub(constraint)
    reconstructionMetrics = InputData.parameterInputFactory("reconstructionMetrics", contentType=InputTypes.StringListType,
                                                            printPriority=108,
                                                            descr=r"""Comma-separated list of native pysensors
                                                                  reconstruction metrics to evaluate on the fitted
                                                                  SSPOR model. Currently supported:
                                                                  \xmlString{rmse}, \xmlString{mse}, and \xmlString{mae}.""")
    goal.addSub(reconstructionMetrics)
    seed = InputData.parameterInputFactory("seed", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The integer seed use for sensor placement random number seed""")
    goal.addSub(seed)
    inputSpecification.addSub(goal)
    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.setInputDataType('xrDataset')
    self.keepInputMeta(False)
    self.outputMultipleRealizations = True                   # True indicate multiple realizations are returned
    self.pivotParameter = None                               # time-dependent data pivot parameter. None if the problem is steady state
    self.validDataType = ['PointSet','HistorySet','DataSet'] # FIXME: Should remove the unsupported ones
    self.sparseSensingGoal = None                            # The goal of the sensor selection. i.e., reconstruction or classification
    self.nSensors = None                                     # The number of the sensors required by the user.
    self.nModes = None                                       # The number of modes/basis used to truncate the singular value decomposition
    self.basis = None                                        # The types of basis used in the projection. i.e., SVD, Identity, or Random Projection
    self.sensingFeatures = None                              # The variable representing the features of the data i.e., X, Y, SensorID, etc.
    self.sensingTarget = None                                # The Response of interest to be reconstructed (or classify)
    self.optimizer = None                                    # The Optimizer type using in the Sparse sensing selection (default: QR)
    self.sensorCosts = None                                  # Optional per-sensor costs for CCQR
    self.sensorCostsVariableName = None                      # Input variable name holding the CCQR costs
    self.constraintSpec = None                               # Optional GQR constraint specification
    self.reconstructionMetrics = []                          # Optional native pysensors reconstruction metrics
    self.seed = None                                         # The seed used by pysensors during sensor selection
    self.sampleTag = 'RAVEN_sample_ID'                       # The sample tag

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the DataClassifier post-processor.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, optional, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if len(inputs)>1:
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only one dataObject')

  def _handleInput(self, paramInput):
    """
      Function to handle the parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    self.name = paramInput.parameterValues['name']
    for child in paramInput.subparts:
      if child.getName() == 'Goal':
        self.sparseSensingGoal = child.parameterValues['subType']
        self.nSensors = child.findFirst('nSensors').value
        self.nModes = child.findFirst('nModes').value
        self.basis = self._normalizeBasis(child.findFirst('basis').value)
        self.sensingFeatures = child.findFirst('features').value
        self.sensingTarget = child.findFirst('target').value
        self.optimizer = self._normalizeOptimizer(child.findFirst('optimizer').value)
        sensorCosts = child.findFirst('sensorCosts')
        self.sensorCostsVariableName = sensorCosts.value if sensorCosts is not None else None
        constraintNode = child.findFirst('constraint')
        self.constraintSpec = self._parseConstraintNode(constraintNode) if constraintNode is not None else None
        metricsNode = child.findFirst('reconstructionMetrics')
        if metricsNode is not None:
          self.reconstructionMetrics = [self._normalizeReconstructionMetric(metricName)
                                        for metricName in metricsNode.value]
        else:
          self.reconstructionMetrics = []
        if child.findFirst('seed') is not None:
          self.seed = child.findFirst('seed').value
        else:
          self.seed = None
        if child.parameterValues['subType'] not in self.goalsDict.keys():
          self.raiseAnError(IOError, '{} is not a recognized option, allowed options are {}'.format(child.getName(),self.goalsDict.keys()))
    goalNode = paramInput.findFirst('Goal')
    _, notFound = goalNode.findNodesAndExtractValues(['nModes','nSensors','features','target'])
    # notFound must be empty
    assert not notFound, "Unexpected nodes in _handleInput"

  def _normalizeBasis(self, basisName):
    """
      Normalize basis names to the canonical spelling used internally.
      @ In, basisName, str, user-provided basis name
      @ Out, normalized, str, canonical basis name
    """
    normalized = self.basisAliases.get(str(basisName).lower())
    if normalized is None:
      self.raiseAnError(IOError, 'basis "{}" is not recognized'.format(basisName))
    return normalized

  def _normalizeOptimizer(self, optimizerName):
    """
      Normalize optimizer names to the canonical spelling used internally.
      @ In, optimizerName, str, user-provided optimizer name
      @ Out, normalized, str, canonical optimizer name
    """
    normalized = self.optimizerAliases.get(str(optimizerName).lower())
    if normalized is None:
      self.raiseAnError(IOError, 'optimizer "{}" is not recognized'.format(optimizerName))
    return normalized

  def _normalizeReconstructionMetric(self, metricName):
    """
      Normalize supported native pysensors reconstruction metric names.
      @ In, metricName, str, user-provided reconstruction metric name
      @ Out, normalized, str, canonical lowercase metric name
    """
    normalized = str(metricName).strip().lower()
    if normalized not in self.reconstructionMetricOptions:
      self.raiseAnError(IOError, 'reconstruction metric "{}" is not recognized; allowed values are {}'.format(metricName, self.reconstructionMetricOptions))
    return normalized

  def _parseConstraintNode(self, constraintNode):
    """
      Parse and validate the optional GQR constraint block.
      @ In, constraintNode, ParameterInput, parsed constraint node
      @ Out, spec, dict, normalized constraint specification
    """
    spec = {'strategy': constraintNode.parameterValues['strategy']}
    for child in constraintNode.subparts:
      if child.getName() == 'vertex':
        spec.setdefault('vertices', []).append(tuple(child.value))
      else:
        spec[child.getName()] = child.value
    if spec['strategy'] == 'distance':
      for key in ['xAxis', 'yAxis', 'radius']:
        if key not in spec:
          self.raiseAnError(IOError, 'GQR distance constraints require <{}>'.format(key))
      if 'shape' in spec or 'indices' in spec:
        self.raiseAnError(IOError, 'GQR distance constraints cannot be combined with shape- or index-based constrained regions')
      return spec
    if 'shape' not in spec and 'indices' not in spec:
      self.raiseAnError(IOError, 'GQR constraints require either <shape> or <indices> unless the strategy is "distance"')
    if spec['strategy'] in ['max_n', 'exact_n', 'predetermined'] and 'nConstSensors' not in spec:
      self.raiseAnError(IOError, 'GQR strategy "{}" requires <nConstSensors>'.format(spec['strategy']))
    if 'indices' in spec:
      return spec
    for key in ['xAxis', 'yAxis']:
      if key not in spec:
        self.raiseAnError(IOError, 'GQR shape constraints require <{}>'.format(key))
    shape = spec['shape']
    if shape == 'Circle':
      for key in ['centerX', 'centerY', 'radius']:
        if key not in spec:
          self.raiseAnError(IOError, 'Circle constraints require <{}>'.format(key))
    elif shape == 'Cylinder':
      for key in ['centerX', 'centerY', 'centerZ', 'radius', 'height', 'zAxis']:
        if key not in spec:
          self.raiseAnError(IOError, 'Cylinder constraints require <{}>'.format(key))
    elif shape == 'Ellipse':
      for key in ['centerX', 'centerY', 'width', 'height']:
        if key not in spec:
          self.raiseAnError(IOError, 'Ellipse constraints require <{}>'.format(key))
    elif shape == 'Line':
      for key in ['x1', 'x2', 'y1', 'y2']:
        if key not in spec:
          self.raiseAnError(IOError, 'Line constraints require <{}>'.format(key))
    elif shape == 'Parabola':
      for key in ['h', 'k', 'a', 'loc']:
        if key not in spec:
          self.raiseAnError(IOError, 'Parabola constraints require <{}>'.format(key))
    elif shape == 'Polygon':
      if len(spec.get('vertices', [])) < 3:
        self.raiseAnError(IOError, 'Polygon constraints require at least three <vertex> entries')
    elif shape == 'UserDefined':
      if 'equation' not in spec and 'file' not in spec:
        self.raiseAnError(IOError, 'UserDefined constraints require either <equation> or <file>')
    return spec

  def _buildBasis(self):
    """
      Construct the configured pysensors basis.
      @ In, None
      @ Out, basis, object, instantiated pysensors basis
    """
    if self.basis == 'SVD':
      return ps.basis.SVD(n_basis_modes=self.nModes)
    if self.basis == 'Identity':
      return ps.basis.Identity(n_basis_modes=self.nModes)
    if self.basis == 'RandomProjection':
      return ps.basis.RandomProjection(n_basis_modes=self.nModes)
    self.raiseAnError(IOError, 'basis "{}" is not recognized'.format(self.basis))

  def _buildOptimizer(self):
    """
      Construct the configured pysensors optimizer.
      @ In, None
      @ Out, optimizer, object, instantiated pysensors optimizer
    """
    if self.optimizer == 'QR':
      return ps.optimizers.QR()
    if self.optimizer == 'CCQR':
      if self.sensorCosts is None:
        self.raiseAnError(IOError, 'CCQR requires sensorCosts to be resolved before building the optimizer')
      return ps.optimizers.CCQR(sensor_costs=self.sensorCosts)
    if self.optimizer == 'GQR':
      return ps.optimizers.GQR()
    if self.optimizer == 'TPGR':
      return ps.optimizers.TPGR(n_sensors=self.nSensors)
    self.raiseAnError(IOError, 'optimizer "{}" is not implemented'.format(self.optimizer))

  def _buildModel(self, basis, optimizer):
    """
      Construct the configured pysensors model.
      @ In, basis, object, instantiated pysensors basis
      @ In, optimizer, object, instantiated pysensors optimizer
      @ Out, model, object, instantiated pysensors sparse sensing model
    """
    if self.sparseSensingGoal == 'reconstruction':
      return ps.SSPOR(basis=basis, n_sensors=self.nSensors, optimizer=optimizer)
    if self.sparseSensingGoal == 'classification':
      self.raiseAnError(NotImplementedError, 'SparseSensing classification is not yet implemented in RAVEN')
    self.raiseAnError(IOError, 'goal "{}" is not recognized'.format(self.sparseSensingGoal))

  def _computeReconstructionMetric(self, model, data, metricName):
    """
      Evaluate a native pysensors reconstruction metric on the fitted model.
      @ In, model, ps.SSPOR, fitted sparse reconstruction model
      @ In, data, np.ndarray, full-state measurements with shape (samples, features)
      @ In, metricName, str, canonical metric name
      @ Out, metricValue, float, scalar metric value
    """
    if metricName == 'rmse':
      # pysensors.score follows sklearn's "higher is better" convention, so RMSE is negated there.
      return float(-model.score(data))
    if metricName == 'mse':
      return float(model.score(data, score_function=lambda yTrue, yPred: np.mean((yTrue - yPred) ** 2)))
    if metricName == 'mae':
      return float(model.score(data, score_function=lambda yTrue, yPred: np.mean(np.abs(yTrue - yPred))))
    self.raiseAnError(IOError, 'reconstruction metric "{}" is not implemented'.format(metricName))

  def _buildConstraintInfoDataFrame(self, inputDS):
    """
      Build a per-sensor dataframe used by pysensors dataframe-based constraint helpers.
      @ In, inputDS, xr.Dataset, input dataset
      @ Out, infoDF, pd.DataFrame, one row per candidate sensor location
    """
    info = {}
    for var in self.sensingFeatures:
      values = np.asarray(inputDS[var].data)
      if values.ndim == 2:
        if var != self.sensingTarget and not np.allclose(values, values[0:1, :]):
          self.raiseAWarning('Feature "{}" varies across samples; using the first sample for GQR constraint geometry'.format(var))
        info[var] = np.asarray(values[0], dtype=float)
      elif values.ndim == 1:
        info[var] = np.asarray(values, dtype=float)
      else:
        self.raiseAnError(IOError, 'Feature "{}" must be 1-D or 2-D for GQR constraints'.format(var))
    targetValues = np.asarray(inputDS[self.sensingTarget].data)
    if targetValues.ndim == 2:
      info[self.sensingTarget] = np.asarray(targetValues[0], dtype=float)
    elif targetValues.ndim == 1:
      info[self.sensingTarget] = np.asarray(targetValues, dtype=float)
    else:
      self.raiseAnError(IOError, 'Target "{}" must be 1-D or 2-D for GQR constraints'.format(self.sensingTarget))
    return pd.DataFrame(info)

  def _computeReferenceSensorRanking(self, data):
    """
      Compute the unconstrained QR ranking used by pysensors GQR helper logic.
      @ In, data, np.ndarray, training data with shape (samples, features)
      @ Out, sensors, np.ndarray, ranked list of all sensor indices
    """
    referenceModel = ps.SSPOR(basis=self._buildBasis(), n_sensors=self.nSensors, optimizer=ps.optimizers.QR())
    if self.seed is not None:
      referenceModel.fit(data, seed=self.seed)
    else:
      referenceModel.fit(data)
    return np.asarray(referenceModel.get_all_sensors(), dtype=int)

  def _buildConstraintObject(self, allSensors, infoDF):
    """
      Build the configured pysensors constraint helper object.
      @ In, allSensors, np.ndarray, ranked list of candidate sensors
      @ In, infoDF, pd.DataFrame, dataframe with per-sensor coordinates
      @ Out, constraint, object, instantiated pysensors constraint helper
    """
    spec = self.constraintSpec
    common = {'data': infoDF,
              'X_axis': spec['xAxis'],
              'Y_axis': spec['yAxis'],
              'Field': self.sensingTarget}
    shape = spec['shape']
    loc = spec.get('loc', 'in')
    if shape == 'Circle':
      return ps.utils.Circle(spec['centerX'], spec['centerY'], spec['radius'], loc=loc, **common)
    if shape == 'Cylinder':
      common['Z_axis'] = spec['zAxis']
      return ps.utils.Cylinder(spec['centerX'], spec['centerY'], spec['centerZ'],
                               spec['radius'], spec['height'], loc=loc, **common)
    if shape == 'Ellipse':
      return ps.utils.Ellipse(spec['centerX'], spec['centerY'], spec['width'],
                              spec['height'], angle=spec.get('angle', 0.0), loc=loc, **common)
    if shape == 'Line':
      return ps.utils.Line(spec['x1'], spec['x2'], spec['y1'], spec['y2'], **common)
    if shape == 'Parabola':
      return ps.utils.Parabola(spec['h'], spec['k'], spec['a'], spec['loc'], **common)
    if shape == 'Polygon':
      return ps.utils.Polygon(spec['vertices'], loc=loc, **common)
    if shape == 'UserDefined':
      if 'file' in spec:
        return ps.utils.UserDefinedConstraints(allSensors, file=spec['file'], **common)
      return ps.utils.UserDefinedConstraints(allSensors, equation=spec['equation'], **common)
    self.raiseAnError(IOError, 'Unsupported GQR constraint shape "{}"'.format(shape))

  def _buildOptimizerKws(self, data, inputDS):
    """
      Build fit-time optimizer kwargs for pysensors.
      @ In, data, np.ndarray, training data with shape (samples, features)
      @ In, inputDS, xr.Dataset, input dataset
      @ Out, optimizerKws, dict, keyword arguments forwarded to model.fit
    """
    if self.optimizer != 'GQR':
      return {}
    if self.constraintSpec is None:
      self.raiseAnError(IOError, 'GQR requires a <constraint> block in SparseSensing')
    allSensors = self._computeReferenceSensorRanking(data)
    optimizerKws = {'all_sensors': allSensors,
                    'n_sensors': self.nSensors,
                    'constraint_option': self.constraintSpec['strategy']}
    if self.constraintSpec['strategy'] == 'distance':
      infoDF = self._buildConstraintInfoDataFrame(inputDS)
      optimizerKws.update({'info': infoDF,
                           'r': self.constraintSpec['radius'],
                           'X_axis': self.constraintSpec['xAxis'],
                           'Y_axis': self.constraintSpec['yAxis']})
      return optimizerKws
    if 'indices' in self.constraintSpec:
      idxConstrained = np.asarray(self.constraintSpec['indices'], dtype=int)
    else:
      infoDF = self._buildConstraintInfoDataFrame(inputDS)
      constraintObject = self._buildConstraintObject(allSensors, infoDF)
      if self.constraintSpec['shape'] == 'UserDefined':
        idxConstrained, _ = constraintObject.constraint()
      else:
        idxConstrained, _ = constraintObject.get_constraint_indices(allSensors, infoDF)
      idxConstrained = np.asarray(idxConstrained, dtype=int)
    optimizerKws['idx_constrained'] = idxConstrained
    optimizerKws['n_const_sensors'] = self.constraintSpec['nConstSensors']
    return optimizerKws

  def run(self,inputIn):
    """
      This method executes the postprocessor action. In this case, it finds the optimal sensor locations to achieve a prescribed goal
      (i.e., reconstruction of a certain response of interest, or classify between data different scenarios)
      @ In, inputIn, dict, dictionaries which contains the data inside the input DataObjects
      @ Out, outputDic, dict, dictionary which contains the data to be collected by output DataObject
    """
    _, _, inputDS = inputIn['Data'][0]

    ## identify features
    self.features = list(self.sensingFeatures)
    # don't keep the pivot parameter in the feature space
    if self.pivotParameter in self.features:
      self.features.remove(self.pivotParameter)
    basis = self._buildBasis()

    features = {}
    for var in self.sensingFeatures:
      features[var] = np.atleast_1d(inputDS[var].data)
    nSamples,nfeatures = np.shape(features[self.sensingFeatures[0]])
    data = inputDS[self.sensingTarget].data
    ## TODO: add some assertions to check the shape of the data matrix in case of steady state and time-dependent data
    assert np.shape(data) == (nSamples,nfeatures)
    if self.sensorCostsVariableName is not None:
      if self.sensorCostsVariableName not in inputDS:
        self.raiseAnError(IOError, 'sensorCosts variable "{}" not found in the input DataObject'.format(self.sensorCostsVariableName))
      rawCosts = np.asarray(inputDS[self.sensorCostsVariableName].data)
      if rawCosts.ndim == 2:
        if not np.allclose(rawCosts, rawCosts[0:1, :]):
          self.raiseAnError(IOError, 'sensorCosts must be invariant across samples for steady-state SparseSensing')
        rawCosts = rawCosts[0]
      elif rawCosts.ndim != 1:
        self.raiseAnError(IOError, 'sensorCosts must be a 1-D vector or a sample-invariant 2-D array')
      self.sensorCosts = np.asarray(rawCosts, dtype=float).reshape(-1)
      if len(self.sensorCosts) != nfeatures:
        self.raiseAnError(IOError, 'sensorCosts has length {} but expected {}'.format(len(self.sensorCosts), nfeatures))
    optimizer = self._buildOptimizer()
    model = self._buildModel(basis, optimizer)
    optimizerKws = self._buildOptimizerKws(data, inputDS)
    if self.seed is not None:
      model.fit(data, seed=self.seed, **optimizerKws)
    else:
      model.fit(data, **optimizerKws)
    selectedSensors = model.get_selected_sensors()
    coords = {'sensor':np.arange(1,len(selectedSensors)+1)}

    sensorData = {}
    for var in self.sensingFeatures:
      sensorData[var] = ('sensor', inputDS[var][0,selectedSensors].data)
    outDS = xr.Dataset(data_vars=sensorData, coords=coords)
    if self.sparseSensingGoal == 'reconstruction' and self.reconstructionMetrics:
      for metricName in self.reconstructionMetrics:
        outDS[metricName] = self._computeReconstructionMetric(model, data, metricName)
    ## PLEASE READ: For developers: this is really important, currently,
    # you have to manually add RAVEN_sample_ID to the dims if you are using xarrays
    outDS = outDS.expand_dims(self.sampleTag)
    outDS[self.sampleTag] = [0]
    return outDS
