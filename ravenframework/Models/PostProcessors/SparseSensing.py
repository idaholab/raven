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
from ... import MetricDistributor

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
    basis = InputData.parameterInputFactory("basis",
                                            contentType=InputTypes.makeEnumType("basis", "basis Type",
                                                                                ['Identity', 'SVD', 'RandomProjection']),
                                            printPriority=108,
                                            descr=r"""The type of basis onto which the data are projected""",
                                            default='SVD')
    goal.addSub(basis)
    nModes = InputData.parameterInputFactory("nModes", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The number of modes retained""")
    goal.addSub(nModes)
    nSensors = InputData.parameterInputFactory("nSensors", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The number of sensors used""")
    goal.addSub(nSensors)
    optimizer = InputData.parameterInputFactory("optimizer",
                                                contentType=InputTypes.makeEnumType("optimizer", "optimizer type",
                                                                                    ['QR', 'CCQR', 'GQR', 'TPGR']),
                                                printPriority=108,
                                                descr=r"""The type of optimizer used. QR: standard QR-pivoting optimizer.
                                                          CCQR: cost-constrained QR, requires sensorCosts.
                                                          GQR: greedy QR.
                                                          TPGR: two-phase greedy.""",
                                                default='QR')
    goal.addSub(optimizer)
    sensorCosts = InputData.parameterInputFactory("sensorCosts", contentType=InputTypes.FloatListType,
                                                  printPriority=108,
                                                  descr=r"""Per-sensor cost vector used by the CCQR optimizer.
                                                            Must have length equal to the total number of candidate sensors.""")
    goal.addSub(sensorCosts)
    seed = InputData.parameterInputFactory("seed", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The integer seed use for sensor placement random number seed""")
    goal.addSub(seed)
    inputSpecification.addSub(goal)

    # Optional Metric assembler nodes (class and type attributes required by RAVEN assembler system).
    # Valid only for reconstruction goal. The metric is evaluated between the reconstructed
    # field (predicted from selected sensor readings) and the original full-field data.
    # Appropriate RAVEN metric types: SKL regression metrics (mean_absolute_error,
    # mean_squared_error, r2_score, explained_variance_score).
    metricInput = InputData.parameterInputFactory("Metric", contentType=InputTypes.StringType,
                                                  printPriority=108,
                                                  descr=r"""Name of a RAVEN Metric object to evaluate reconstruction error.
                                                            Must have attributes class='Metrics' and type='Metric'.
                                                            Can be repeated for multiple metrics. Only valid for reconstruction goal.""")
    metricInput.addParam("class", InputTypes.StringType, True)
    metricInput.addParam("type", InputTypes.StringType, True)
    inputSpecification.addSub(metricInput)

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
    self.validDataType = ['PointSet','HistorySet','DataSet'] # DataSet required when reconstruction metrics are requested
    self.sparseSensingGoal = None                            # The goal of the sensor selection. i.e., reconstruction or classification
    self.nSensors = None                                     # The number of the sensors required by the user.
    self.nModes = None                                       # The number of modes/basis used to truncate the singular value decomposition
    self.basis = None                                        # The types of basis used in the projection. i.e., SVD, Identity, or Random Projection
    self.sensingFeatures = None                              # The variable representing the features of the data i.e., X, Y, SensorID, etc.
    self.sensingTarget = None                                # The Response of interest to be reconstructed (or classify)
    self.optimizer = None                                    # The Optimizer type using in the Sparse sensing selection (default: QR)
    self.sensorCosts = None                                  # Optional cost vector for CCQR optimizer
    self.sampleTag = 'RAVEN_sample_ID'                       # The sample tag
    self.metricsDict = {}                                    # assembled Metric objects {name: instance}
    # Register optional Metric assembler objects
    self.addAssemblerObject('Metric', InputData.Quantity.zero_to_infinity)

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the DataClassifier post-processor.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, optional, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    if len(inputs) > 1:
      self.raiseAnError(IOError, 'Post-Processor', self.name, 'accepts only one dataObject')
    # Retrieve assembled Metric objects
    for metricIn in self.assemblerDict.get('Metric', []):
      self.metricsDict[metricIn[2]] = metricIn[3]
    if self.metricsDict and self.sparseSensingGoal != 'reconstruction':
      self.raiseAWarning('Metric objects are provided but goal is not "reconstruction". Metrics will be ignored.')

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
        self.basis = child.findFirst('basis').value
        self.sensingFeatures = child.findFirst('features').value
        self.sensingTarget = child.findFirst('target').value
        self.optimizer = child.findFirst('optimizer').value
        seedNode = child.findFirst('seed')
        self.seed = seedNode.value if seedNode is not None else None
        costsNode = child.findFirst('sensorCosts')
        self.sensorCosts = np.asarray(costsNode.value) if costsNode is not None else None
        if self.sparseSensingGoal not in self.goalsDict:
          self.raiseAnError(IOError, '{} is not a recognized option, allowed options are {}'.format(
            child.getName(), list(self.goalsDict.keys())))
      elif child.getName() == 'Metric':
        if 'type' not in child.parameterValues or 'class' not in child.parameterValues:
          self.raiseAnError(IOError, 'Tag Metric must have attributes "class" and "type"')
    if self.sparseSensingGoal is None:
      self.raiseAnError(IOError, 'A <Goal> sub-element is required in SparseSensing PostProcessor')
    _, notFound = paramInput.subparts[0].findNodesAndExtractValues(['nModes','nSensors','features','target'])
    assert not notFound, "Unexpected nodes in _handleInput"

  def _buildBasis(self):
    """
      Build the pysensors basis object from the parsed configuration.
      @ In, None
      @ Out, basis, pysensors basis instance
    """
    basisLow = self.basis.lower()
    if basisLow == 'svd':
      return ps.basis.SVD(n_basis_modes=self.nModes)
    elif basisLow == 'identity':
      return ps.basis.Identity(n_basis_modes=self.nModes)
    elif basisLow == 'randomprojection':
      return ps.basis.RandomProjection(n_basis_modes=self.nModes)
    else:
      self.raiseAnError(IOError, 'Basis type "{}" is not recognized'.format(self.basis))

  def _buildOptimizer(self):
    """
      Build the pysensors optimizer object from the parsed configuration.
      @ In, None
      @ Out, optimizer, pysensors optimizer instance
    """
    optLow = self.optimizer.lower()
    if optLow == 'qr':
      return ps.optimizers.QR()
    elif optLow == 'ccqr':
      if self.sensorCosts is None:
        self.raiseAnError(IOError, 'CCQR optimizer requires <sensorCosts> to be specified')
      return ps.optimizers.CCQR(sensor_costs=self.sensorCosts)
    elif optLow == 'gqr':
      return ps.optimizers.GQR()
    elif optLow == 'tpgr':
      return ps.optimizers.TPGR(n_sensors=self.nSensors)
    else:
      self.raiseAnError(IOError, 'Optimizer type "{}" is not implemented'.format(self.optimizer))

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it finds the optimal sensor locations to achieve a prescribed goal
      (i.e., reconstruction of a certain response of interest, or classify between data different scenarios)
      @ In, inputIn, dict, dictionaries which contains the data inside the input DataObjects
      @ Out, outDS, xr.Dataset, dataset containing selected sensor locations and (optionally) reconstruction error metrics
    """
    _, _, inputDS = inputIn['Data'][0]

    ## identify features
    self.features = list(self.sensingFeatures)
    # don't keep the pivot parameter in the feature space
    if self.pivotParameter in self.features:
      self.features.remove(self.pivotParameter)

    basis = self._buildBasis()
    optimizer = self._buildOptimizer()

    features = {}
    for var in self.sensingFeatures:
      features[var] = np.atleast_1d(inputDS[var].data)
    nSamples, nfeatures = np.shape(features[self.sensingFeatures[0]])
    data = inputDS[self.sensingTarget].data
    ## TODO: add some assertions to check the shape of the data matrix in case of steady state and time-dependent data
    assert np.shape(data) == (nSamples, nfeatures)

    if self.sparseSensingGoal == 'reconstruction':
      model = ps.SSPOR(basis=basis, n_sensors=self.nSensors, optimizer=optimizer)
    else:
      model = ps.SSPOC(basis=basis, n_sensors=self.nSensors, optimizer=optimizer)

    if self.seed is not None:
      model.fit(data, seed=self.seed)
    else:
      model.fit(data)

    selectedSensors = model.get_selected_sensors()
    coords = {'sensor': np.arange(1, len(selectedSensors) + 1)}

    sensorData = {}
    for var in self.sensingFeatures:
      sensorData[var] = ('sensor', inputDS[var][0, selectedSensors].data)
    outDS = xr.Dataset(data_vars=sensorData, coords=coords)

    # Compute reconstruction error metrics (SSPOR only)
    if self.sparseSensingGoal == 'reconstruction' and self.metricsDict:
      # Reconstruct the full field from sensor readings at selected locations
      sensorReadings = data[:, selectedSensors]           # shape (nSamples, nSensors)
      reconstructed = model.predict(sensorReadings)       # shape (nSamples, nfeatures)
      weights = np.ones(nSamples)
      pairedData = ((reconstructed, weights), (data, weights))
      for metricName, metricInstance in self.metricsDict.items():
        metricEngine = MetricDistributor.factory.returnInstance('MetricDistributor', metricInstance)
        errorValue = metricEngine.evaluate(pairedData, weights=None, multiOutput='mean')
        # errorValue is a numpy array of shape (1,); store as a scalar variable
        outDS[metricName] = float(errorValue[0])

    ## PLEASE READ: For developers: this is really important, currently,
    # you have to manually add RAVEN_sample_ID to the dims if you are using xarrays
    outDS = outDS.expand_dims(self.sampleTag)
    outDS[self.sampleTag] = [0]
    return outDS
