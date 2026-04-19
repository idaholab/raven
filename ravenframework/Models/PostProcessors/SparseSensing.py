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
  optimizerOptions = ['QR', 'CCQR']
  optimizerAliases = {'qr': 'QR',
                      'ccqr': 'CCQR'}
  reconstructionMetricOptions = ['rmse', 'mse', 'mae']

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
    if self.seed is not None:
      model.fit(data, seed=self.seed)
    else:
      model.fit(data)
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
