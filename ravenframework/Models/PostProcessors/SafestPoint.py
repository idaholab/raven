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
Created on July 10, 2013

@author: alfoa
"""
#External Modules------------------------------------------------------------------------------------
import numpy as np
import xarray
from scipy import spatial
import copy
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from .BasicStatistics import BasicStatistics
from ...utils import InputData, InputTypes
from ...utils.RAVENiterators import ravenArrayIterator
#Internal Modules End--------------------------------------------------------------------------------

class SafestPoint(PostProcessorInterface):
  """
    It searches for the probability-weighted safest point inside the space of the system controllable variables
  """
  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    inputSpecification = super(SafestPoint, cls).getInputSpecification()

    OuterDistributionInput = InputData.parameterInputFactory("Distribution", contentType=InputTypes.StringType)
    OuterDistributionInput.addParam("class", InputTypes.StringType)
    OuterDistributionInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(OuterDistributionInput)

    VariableInput = InputData.parameterInputFactory("variable")
    VariableInput.addParam("name", InputTypes.StringType)
    InnerDistributionInput = InputData.parameterInputFactory("distribution", contentType=InputTypes.StringType)
    VariableInput.addSub(InnerDistributionInput)
    InnerGridInput = InputData.parameterInputFactory("grid", contentType=InputTypes.FloatType)
    InnerGridInput.addParam("type", InputTypes.StringType)
    InnerGridInput.addParam("steps", InputTypes.IntegerType)
    VariableInput.addSub(InnerGridInput)
    ControllableInput = InputData.parameterInputFactory("controllable", contentType=InputTypes.StringType)
    ControllableInput.addSub(VariableInput)
    inputSpecification.addSub(ControllableInput)
    inputSpecification.addSub(InputData.parameterInputFactory("outputName", contentType=InputTypes.StringType))

    NoncontrollableInput = InputData.parameterInputFactory("non-controllable", contentType=InputTypes.StringType)
    NoncontrollableInput.addSub(VariableInput)
    inputSpecification.addSub(NoncontrollableInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, runInfoDict, dict, run info
      @ Out, None
    """
    super().__init__()
    # delay loading for import order
    from ...Models.PostProcessors import factory
    self.controllableDist = {}  # dictionary created upon the .xml input file reading. It stores the distributions for each controllable variable.
    self.nonControllableDist = {}  # dictionary created upon the .xml input file reading. It stores the distributions for each non-controllable variable.
    self.controllableGrid = {}  # dictionary created upon the .xml input file reading. It stores the grid type ('value' or 'CDF'), the number of steps and the step length for each controllale variable.
    self.nonControllableGrid = {}  # dictionary created upon the .xml input file reading. It stores the grid type ('value' or 'CDF'), the number of steps and the step length for each non-controllale variable.
    self.gridInfo = {}  # dictionary containing the grid type ('value' or 'CDF'), the grid construction type ('equal', set by default) and the list of sampled points for each variable.
    self.controllableOrd = []  # list containing the controllable variables' names in the same order as they appear inside the controllable space (self.controllableSpace)
    self.nonControllableOrd = []  # list containing the controllable variables' names in the same order as they appear inside the non-controllable space (self.nonControllableSpace)
    self.surfPointsMatrix = None  # 2D-matrix containing the coordinates of the points belonging to the failure boundary (coordinates are derived from both the controllable and non-controllable space)
    self.stat = factory.returnInstance('BasicStatistics')  # instantiation of the 'BasicStatistics' processor, which is used to compute the expected value of the safest point through the coordinates and probability values collected in the 'run' function
    self.outputName = "Probability"
    self.addAssemblerObject('Distribution', InputData.Quantity.one_to_infinity)
    self.addMetaKeys(["ProbabilityWeight"])
    self.printTag = 'POSTPROCESSOR SAFESTPOINT'

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName() == 'outputName':
        self.outputName = child.value
      if child.getName() == 'controllable' or  child.getName() == 'non-controllable':
        for childChild in child.subparts:
          if childChild.getName() == 'variable':
            varName = childChild.parameterValues['name']
            for childChildChild in childChild.subparts:
              if childChildChild.getName() == 'distribution':
                if child.getName() == 'controllable':
                  self.controllableDist[varName] = childChildChild.value
                elif child.getName() == 'non-controllable':
                  self.nonControllableDist[varName] = childChildChild.value
              elif childChildChild.getName() == 'grid':
                if 'type' in childChildChild.parameterValues:
                  if 'steps' in childChildChild.parameterValues:
                    childChildInfo = (childChildChild.parameterValues['type'], childChildChild.parameterValues['steps'], childChildChild.value)
                    if child.getName() == 'controllable':
                      self.controllableGrid[varName] = childChildInfo
                    elif child.getName() == 'non-controllable':
                      self.nonControllableGrid[varName] = childChildInfo
                  else:
                    self.raiseAnError(NameError, 'number of steps missing after the grid call.')
                else:
                  self.raiseAnError(NameError, 'grid type missing after the grid call.')
              else:
                self.raiseAnError(NameError, 'invalid labels after the variable call. Only "distribution" and "grid" are accepted.')
          else:
            self.raiseAnError(NameError, 'invalid or missing labels after the '+child.getName()+' variables call. Only "variable" is accepted.')
    self.raiseADebug('CONTROLLABLE DISTRIBUTIONS:')
    self.raiseADebug(self.controllableDist)
    self.raiseADebug('CONTROLLABLE GRID:')
    self.raiseADebug(self.controllableGrid)
    self.raiseADebug('NON-CONTROLLABLE DISTRIBUTIONS:')
    self.raiseADebug(self.nonControllableDist)
    self.raiseADebug('NON-CONTROLLABLE GRID:')
    self.raiseADebug(self.nonControllableGrid)

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the Safest Point pp. This method is in charge
      of creating the Controllable and no-controllable grid.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    for varName, distName in self.controllableDist.items():
      self.controllableDist[varName] = self.retrieveObjectFromAssemblerDict('Distribution', distName)
    for varName, distName in self.nonControllableDist.items():
      self.nonControllableDist[varName] = self.retrieveObjectFromAssemblerDict('Distribution', distName)
    self.__gridSetting__()
    self.__gridGeneration__()
    self.inputToInternal(inputs)
    #FIXME this is quite invasive use of the basic statistics; a more standardized construction would be nice
    #we set the toDo here, since at this point we know the targets for the basic statistics
    self.stat.toDo = {'expectedValue':[{'targets':set(self.controllableOrd), 'prefix':"controllable"}]} #don't set directly, just set up the toDo for basicStats
    self.stat.initialize(runInfo, inputs, initDict)
    self.raiseADebug('GRID INFO:')
    self.raiseADebug(self.gridInfo)
    self.raiseADebug('N-DIMENSIONAL CONTROLLABLE SPACE:')
    self.raiseADebug(self.controllableSpace)
    self.raiseADebug('N-DIMENSIONAL NON-CONTROLLABLE SPACE:')
    self.raiseADebug(self.nonControllableSpace)
    self.raiseADebug('CONTROLLABLE VARIABLES ORDER:')
    self.raiseADebug(self.controllableOrd)
    self.raiseADebug('NON-CONTROLLABLE VARIABLES ORDER:')
    self.raiseADebug(self.nonControllableOrd)
    self.raiseADebug('SURFACE POINTS MATRIX:')
    self.raiseADebug(self.surfPointsMatrix)

  def __gridSetting__(self, constrType = 'equal'):
    """
      Set up the grid
      @ In, constrType, string, optional, the type of grid to construct (equal,custom)
      @ Out, None
    """
    for varName in self.controllableGrid.keys():
      if self.controllableGrid[varName][0] == 'value':
        self.__stepError__(float(self.controllableDist[varName].lowerBound), float(self.controllableDist[varName].upperBound), self.controllableGrid[varName][1], self.controllableGrid[varName][2], varName)
        self.gridInfo[varName] = (self.controllableGrid[varName][0], constrType, [float(self.controllableDist[varName].lowerBound) + self.controllableGrid[varName][2] * i for i in range(self.controllableGrid[varName][1] + 1)])
      elif self.controllableGrid[varName][0] == 'CDF':
        self.__stepError__(0, 1, self.controllableGrid[varName][1], self.controllableGrid[varName][2], varName)
        self.gridInfo[varName] = (self.controllableGrid[varName][0], constrType, [self.controllableGrid[varName][2] * i for i in range(self.controllableGrid[varName][1] + 1)])
      else:
        self.raiseAnError(NameError, 'inserted invalid grid type. Only "value" and "CDF" are accepted.')
    for varName in self.nonControllableGrid.keys():
      if self.nonControllableGrid[varName][0] == 'value':
        self.__stepError__(float(self.nonControllableDist[varName].lowerBound), float(self.nonControllableDist[varName].upperBound), self.nonControllableGrid[varName][1], self.nonControllableGrid[varName][2], varName)
        self.gridInfo[varName] = (self.nonControllableGrid[varName][0], constrType, [float(self.nonControllableDist[varName].lowerBound) + self.nonControllableGrid[varName][2] * i for i in range(self.nonControllableGrid[varName][1] + 1)])
      elif self.nonControllableGrid[varName][0] == 'CDF':
        self.__stepError__(0, 1, self.nonControllableGrid[varName][1], self.nonControllableGrid[varName][2], varName)
        self.gridInfo[varName] = (self.nonControllableGrid[varName][0], constrType, [self.nonControllableGrid[varName][2] * i for i in range(self.nonControllableGrid[varName][1] + 1)])
      else:
        self.raiseAnError(NameError, 'inserted invalid grid type. Only "value" and "CDF" are accepted.')

  def __stepError__(self, lowerBound, upperBound, steps, tol, varName):
    """
      Method to check if the lowerBound and upperBound are not consistent with the tol and stepsize
      @ In, lowerBound, float, lower bound
      @ In, upperBound, float, upper bound
      @ In, steps, int, number of steps
      @ In, tol, float, grid tolerance
      @ In, varName, string, variable name
      @ Out, None
    """
    if upperBound - lowerBound < steps * tol:
      self.raiseAnError(IOError, 'requested number of steps or tolerance for variable ' + varName + ' exceeds its limit.')

  def __gridGeneration__(self):
    """
      Method to generate the grid
      @ In, None
      @ Out, None
    """
    NotchesByVar = [None] * len(self.controllableGrid.keys())
    controllableSpaceSize = None
    for varId, varName in enumerate(self.controllableGrid.keys()):
      NotchesByVar[varId] = self.controllableGrid[varName][1] + 1
      self.controllableOrd.append(varName)
    controllableSpaceSize = tuple(NotchesByVar + [len(self.controllableGrid.keys())])
    self.controllableSpace = np.zeros(controllableSpaceSize)
    iterIndex = ravenArrayIterator(arrayIn=self.controllableSpace)
    while not iterIndex.finished:
      coordIndex = iterIndex.multiIndex[-1]
      varName = list(self.controllableGrid.keys())[coordIndex]
      notchPos = iterIndex.multiIndex[coordIndex]
      if self.gridInfo[varName][0] == 'CDF':
        valList = []
        for probVal in self.gridInfo[varName][2]:
          valList.append(self.controllableDist[varName].cdf(probVal))
        self.controllableSpace[iterIndex.multiIndex] = valList[notchPos]
      else:
        self.controllableSpace[iterIndex.multiIndex] = self.gridInfo[varName][2][notchPos]
      iterIndex.iternext()
    NotchesByVar = [None] * len(self.nonControllableGrid.keys())
    nonControllableSpaceSize = None
    for varId, varName in enumerate(self.nonControllableGrid.keys()):
      NotchesByVar[varId] = self.nonControllableGrid[varName][1] + 1
      self.nonControllableOrd.append(varName)
    nonControllableSpaceSize = tuple(NotchesByVar + [len(self.nonControllableGrid.keys())])
    self.nonControllableSpace = np.zeros(nonControllableSpaceSize)
    iterIndex = ravenArrayIterator(arrayIn=self.nonControllableSpace)
    while not iterIndex.finished:
      coordIndex = iterIndex.multiIndex[-1]
      varName = list(self.nonControllableGrid.keys())[coordIndex]
      notchPos = iterIndex.multiIndex[coordIndex]
      if self.gridInfo[varName][0] == 'CDF':
        valList = []
        for probVal in self.gridInfo[varName][2]:
          valList.append(self.nonControllableDist[varName].cdf(probVal))
        self.nonControllableSpace[iterIndex.multiIndex] = valList[notchPos]
      else:
        self.nonControllableSpace[iterIndex.multiIndex] = self.gridInfo[varName][2][notchPos]
      iterIndex.iternext()

  def inputToInternal(self, currentInput):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, None, the resulting converted object is stored as an attribute of this class
    """
    if len(currentInput) > 1:
      self.raiseAnError(IOError,"This PostProcessor can accept only a single input! Got: "+ str(len(currentInput))+"!")
    item = currentInput[0]
    if item.type != 'PointSet':
      self.raiseAnError(IOError, self.type +" accepts PointSet only as input! Got: "+item.type)
    self.surfPointsMatrix = np.zeros((len(item), len(self.gridInfo.keys()) + 1))
    dataSet = item.asDataset()
    for k, varName in enumerate(self.controllableOrd+self.nonControllableOrd):
      self.surfPointsMatrix[:, k] = dataSet[varName].values
    self.surfPointsMatrix[:, k+1] = dataSet[item.getVars("output")[-1]].values

  def run(self, input):
    """
      This method executes the postprocessor action. In this case, it computes the safest point
      @ In,  input, object, object contained the data to process. (inputToInternal output)
      @ Out, dataCollector, PointSet, PointSet containing the elaborated data
    """
    ##TODO: This PP can be LARGELY optimized.
    nearestPointsInd = []
    surfTree = spatial.KDTree(copy.copy(self.surfPointsMatrix[:, 0:self.surfPointsMatrix.shape[-1] - 1]))
    self.controllableSpace.shape = (np.prod(self.controllableSpace.shape[0:len(self.controllableSpace.shape) - 1]), self.controllableSpace.shape[-1])
    self.nonControllableSpace.shape = (np.prod(self.nonControllableSpace.shape[0:len(self.nonControllableSpace.shape) - 1]), self.nonControllableSpace.shape[-1])
    self.raiseADebug('RESHAPED CONTROLLABLE SPACE:')
    self.raiseADebug(self.controllableSpace)
    self.raiseADebug('RESHAPED NON-CONTROLLABLE SPACE:')
    self.raiseADebug(self.nonControllableSpace)
    # create space for realization
    rlz = {key : np.zeros(self.nonControllableSpace.shape[0])  for key in self.controllableOrd+self.nonControllableOrd}
    rlz[self.outputName] = np.zeros(self.nonControllableSpace.shape[0])
    rlz['ProbabilityWeight'] = np.zeros(self.nonControllableSpace.shape[0])
    for ncLine in range(self.nonControllableSpace.shape[0]):
      queryPointsMatrix = np.append(self.controllableSpace, np.tile(self.nonControllableSpace[ncLine, :], (self.controllableSpace.shape[0], 1)), axis = 1)
      self.raiseADebug('QUERIED POINTS MATRIX:')
      self.raiseADebug(queryPointsMatrix)
      nearestPointsInd = surfTree.query(queryPointsMatrix)[-1]
      distList = []
      indexList = []
      probList = []
      for index in range(len(nearestPointsInd)):
        if self.surfPointsMatrix[np.where(np.prod(surfTree.data[nearestPointsInd[index], 0:
          self.surfPointsMatrix.shape[-1] - 1] == self.surfPointsMatrix[:, 0:self.surfPointsMatrix.shape[-1] - 1], axis = 1))[0][0], -1] == 1:
          distList.append(np.sqrt(np.sum(np.power(queryPointsMatrix[index, 0:self.controllableSpace.shape[-1]] - surfTree.data[nearestPointsInd[index], 0:self.controllableSpace.shape[-1]], 2))))
          indexList.append(index)
      if distList == []:
        self.raiseAnError(ValueError, 'no safest point found for the current set of non-controllable variables: ' + str(self.nonControllableSpace[ncLine, :]) + '.')
      else:
        for cVarIndex in range(len(self.controllableOrd)):
          rlz[self.controllableOrd[cVarIndex]][ncLine] = copy.copy(queryPointsMatrix[indexList[distList.index(max(distList))], cVarIndex])
        for ncVarIndex in range(len(self.nonControllableOrd)):
          rlz[self.nonControllableOrd[ncVarIndex]][ncLine] = queryPointsMatrix[indexList[distList.index(max(distList))], len(self.controllableOrd) + ncVarIndex]
          if queryPointsMatrix[indexList[distList.index(max(distList))], len(self.controllableOrd) + ncVarIndex] == self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].lowerBound:
            if self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][0] == 'CDF':
              prob = self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / 2.
            else:
              prob = self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].lowerBound + self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / 2.)
          elif queryPointsMatrix[indexList[distList.index(max(distList))], len(self.controllableOrd) + ncVarIndex] == self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].upperBound:
            if self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][0] == 'CDF':
              prob = self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / 2.
            else:
              prob = 1 - self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].upperBound - self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / 2.)
          else:
            if self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][0] == 'CDF':
              prob = self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2]
            else:
              prob = self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(queryPointsMatrix[indexList[distList.index(max(distList))],
                      len(self.controllableOrd) + ncVarIndex] + self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / 2.) - self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(queryPointsMatrix[indexList[distList.index(max(distList))],
                      len(self.controllableOrd) + ncVarIndex] - self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / 2.)
          probList.append(prob)
      rlz[self.outputName][ncLine] = np.prod(probList)
      rlz['ProbabilityWeight'][ncLine] = np.prod(probList)
    metadata = {'ProbabilityWeight':xarray.DataArray(rlz['ProbabilityWeight'])}
    targets = {tar:xarray.DataArray( rlz[tar])  for tar in self.controllableOrd}
    rlz['ExpectedSafestPointCoordinates'] = self.stat.run({'metadata':metadata, 'targets':targets})
    self.raiseADebug(rlz['ExpectedSafestPointCoordinates'])
    return rlz

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    dataCollector = evaluation[1]

    if output.type != 'PointSet':
      self.raiseAnError(TypeError, 'output item type must be "PointSet".')

    if len(output) > 0:
      self.raiseAnError(ValueError, 'output item must be empty.')

    if self.outputName not in output.getVars():
      self.raiseAnError(IOError, 'The output name "'+self.outputName+'" is not present in the output '+output.name )
    safestPoint = dataCollector.pop("ExpectedSafestPointCoordinates")
    # add the data
    output.load(dataCollector,'dict')
    # add general metadata
    output.addMeta(self.type,{'safest_point':{'coordinate':safestPoint}})
