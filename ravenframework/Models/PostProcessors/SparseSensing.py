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
      if child.parameterValues['subType'] not in self.goalsDict.keys():
        self.raiseAnError(IOError, '{} is not a recognized option, allowed options are {}'.format(child.getName(),self.goalsDict.keys()))
    _, notFound = paramInput.subparts[0].findNodesAndExtractValues(['nModes','nSensors','features','target'])
    # notFound must be empty
    assert not notFound, "Unexpected nodes in _handleInput"

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

    features = {}
    for var in self.sensingFeatures:
      features[var] = np.atleast_1d(inputDS[var].data)
    nSamples,nfeatures = np.shape(features[self.sensingFeatures[0]])
    data = inputDS[self.sensingTarget].data
    ## TODO: add some assertions to check the shape of the data matrix in case of steady state and time-dependent data
    assert np.shape(data) == (nSamples,nfeatures)
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
    ## PLEASE READ: For developers: this is really important, currently,
    # you have to manually add RAVEN_sample_ID to the dims if you are using xarrays
    outDS = outDS.expand_dims(self.sampleTag)
    outDS[self.sampleTag] = [0]
    return outDS
