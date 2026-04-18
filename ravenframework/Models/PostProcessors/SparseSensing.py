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
    inputSpecification.addSub(goal)
    features = InputData.parameterInputFactory("features", contentType=InputTypes.StringListType,
                                                printPriority=108,
                                                descr=r"""Features/inputs of the data model""")
    goal.addSub(features)
    target = InputData.parameterInputFactory("target", contentType=InputTypes.StringType,
                                                printPriority=108,
                                                descr=r"""target of data model""")
    goal.addSub(target)
    basis = InputData.parameterInputFactory("basis", contentType=InputTypes.makeEnumType("basis","basis Type",['Identity','SVD','RandomProjetion']),
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
    optimizer = InputData.parameterInputFactory("optimizer", contentType=InputTypes.makeEnumType("optimizer","optimizer type",['QR']),
                                                           printPriority=108,
                                                           descr=r"""The type of optimizer used""",default='QR')
    goal.addSub(optimizer)
    seed = InputData.parameterInputFactory("seed", contentType=InputTypes.IntegerType,
                                                           printPriority=108,
                                                           descr=r"""The integer seed use for sensor placement random number seed""")
    goal.addSub(seed)
    pivotParameter = InputData.parameterInputFactory('pivotParameter',
                        contentType=InputTypes.StringType, printPriority=108,
                        descr=r"""Name of the pivot dimension in the input data (e.g. 'time'). """
                              r"""When supplied, the data is treated as time-dependent.""")
    goal.addSub(pivotParameter)
    reshape = InputData.parameterInputFactory('reshape',
                        contentType=InputTypes.makeEnumType('reshape','reshapeType',
                                                            ['snapshot','spatiotemporal']),
                        printPriority=108,
                        descr=r"""How to flatten a parameter/time tensor before sensor selection. """
                              r"""'snapshot': stack (sample,time) pairs as rows (sensors = spatial points). """
                              r"""'spatiotemporal': stack (space,time) pairs as columns (sensors = space-time pairs).""",
                        default='snapshot')
    goal.addSub(reshape)
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
    self.reshape = 'snapshot'                                # 'snapshot' | 'spatiotemporal'
    self.validDataType = ['PointSet','HistorySet','DataSet'] # FIXME: Should remove the unsupported ones
    self.sparseSensingGoal = None                            # The goal of the sensor selection. i.e., reconstruction or classification
    self.nSensors = None                                     # The number of the sensors required by the user.
    self.nModes = None                                       # The number of modes/basis used to truncate the singular value decomposition
    self.basis = None                                        # The types of basis used in the projection. i.e., SVD, Identity, or Random Projection
    self.sensingFeatures = None                              # The variable representing the features of the data i.e., X, Y, SensorID, etc.
    self.sensingTarget = None                                # The Response of interest to be reconstructed (or classify)
    self.optimizer = None                                    # The Optimizer type using in the Sparse sensing selection (default: QR)
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
      self.sparseSensingGoal = child.parameterValues['subType']
      self.nSensors = child.findFirst('nSensors').value
      self.nModes = child.findFirst('nModes').value
      self.basis = child.findFirst('basis').value
      self.sensingFeatures = child.findFirst('features').value
      self.sensingTarget = child.findFirst('target').value
      self.optimizer = child.findFirst('optimizer').value
      if child.findFirst('seed') is not None:
        self.seed = child.findFirst('seed').value
      else:
        self.seed = None
      pivot = child.findFirst('pivotParameter')
      self.pivotParameter = pivot.value if pivot is not None else None
      reshape = child.findFirst('reshape')
      self.reshape = reshape.value if reshape is not None else 'snapshot'
      if child.parameterValues['subType'] not in self.goalsDict.keys():
        self.raiseAnError(IOError, '{} is not a recognized option, allowed options are {}'.format(child.getName(),self.goalsDict.keys()))
    _, notFound = paramInput.subparts[0].findNodesAndExtractValues(['nModes','nSensors','features','target'])
    # notFound must be empty
    assert not notFound, "Unexpected nodes in _handleInput"

  def _reshapeForFit(self, data, pivotLen):
    """Reshape a (sample, [time,] space) array into the 2-D matrix SSPOR.fit expects.

    @ In, data, np.ndarray, shape (nSamples, nSpace) or (nSamples, nTime, nSpace).
    @ In, pivotLen, int or None, length of the time axis if present.
    @ Out, matrix, np.ndarray, 2-D matrix (rows=snapshots or samples, cols=sensor candidates).
    """
    if pivotLen is None or data.ndim == 2:
      return data
    if self.reshape == 'snapshot':
      nSamples, nTime, nSpace = data.shape
      # Row-major stack: row k·T + t holds sample k at time t.
      return data.reshape(nSamples * nTime, nSpace)
    raise NotImplementedError(f"reshape={self.reshape} not yet implemented")

  def run(self,inputIn):
    """
      This method executes the postprocessor action. In this case, it finds the optimal sensor locations to achieve a prescribed goal
      (i.e., reconstruction of a certain response of interest, or classify between data different scenarios)
      @ In, inputIn, dict, dictionaries which contains the data inside the input DataObjects
      @ Out, outputDic, dict, dictionary which contains the data to be collected by output DataObject
    """
    _, _, inputDS = inputIn['Data'][0]

    ## identify features
    self.features = self.sensingFeatures
    # don't keep the pivot parameter in the feature space
    if self.pivotParameter in self.features:
      self.features.remove(self.pivotParameter)
    if self.basis.lower() == 'svd':
      basis=ps.basis.SVD(n_basis_modes=self.nModes)
    elif self.basis.lower() == 'identity':
      basis=ps.basis.Identity(n_basis_modes=self.nModes)
    elif self.basis.lower() == 'randomprojection':
      basis=ps.basis.RandomProjection(n_basis_modes=self.nModes)
    else:
      self.raiseAnError(IOError, 'basis are not recognized')

    if self.optimizer.lower() == 'qr':
      optimizer = ps.optimizers.QR()
    else:
      self.raiseAnError(IOError, 'optimizer {} not implemented!!!'.format(self.optimizer))

    model = ps.SSPOR(basis=basis,n_sensors = self.nSensors,optimizer = optimizer)

    data = inputDS[self.sensingTarget].data

    pivotLen = None
    if self.pivotParameter is not None:
      if self.pivotParameter not in inputDS.dims:
        self.raiseAnError(IOError,
          f"pivotParameter '{self.pivotParameter}' not found in input dims {list(inputDS.dims)}")
      pivotLen = inputDS.sizes[self.pivotParameter]

    # Expected shapes:
    #   steady-state (pivotParameter=None): (nSamples, nSpace)
    #   transient / param+time:              (nSamples, nTime, nSpace)
    # Data layout contract (confirmed via Task 1 probe on testSPSLOptiTwist):
    #   - inputDS.dims == {'RAVEN_sample_ID': 4, 'index': 4051}
    #   - inputDS[target].dims == ('RAVEN_sample_ID', 'index')
    #   - shape == (nSamples, nPointsAlongPivot) = (4, 4051)
    # When <pivotParameter> is NOT declared (current OPTI-TWIST case):
    #   the pivot dim holds spatial indices; matrix is ready for SSPOR as-is.
    # When <pivotParameter> IS declared (new transient/parametric cases, future tasks):
    #   the pivot dim holds time; spatial dim comes from a separate feature axis
    #   and we must reshape (see _reshapeForFit).
    if pivotLen is None:
      assert data.ndim == 2, f"Expected 2-D target for steady-state; got {data.shape}"
      nSamples, nSpace = data.shape
    else:
      assert data.ndim == 3, f"Expected 3-D target when pivotParameter is set; got {data.shape}"
      nSamples, _nTime, nSpace = data.shape

    matrix = self._reshapeForFit(data, pivotLen)
    if self.seed is not None:
      model.fit(matrix, seed=self.seed)
    else:
      model.fit(matrix)
    selectedSensors = model.get_selected_sensors()
    coords = {'sensor':np.arange(1,len(selectedSensors)+1)}

    sensorData = {}
    for var in self.sensingFeatures:
      arr = inputDS[var].data
      # Reduce arr to a (nSpace,) vector: drop sample axis (take index 0) and,
      # when present, drop the pivot axis too (features are assumed space-only).
      if arr.ndim == 2:
        vec = arr[0, :]
      elif arr.ndim == 3:
        vec = arr[0, 0, :]
      else:
        vec = np.atleast_1d(arr)
      sensorData[var] = ('sensor', vec[selectedSensors])
    outDS = xr.Dataset(data_vars=sensorData, coords=coords)
    ## PLEASE READ: For developers: this is really important, currently,
    # you have to manually add RAVEN_sample_ID to the dims if you are using xarrays
    outDS = outDS.expand_dims(self.sampleTag)
    outDS[self.sampleTag] = [0]
    return outDS
