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
import copy
from collections import OrderedDict
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from ...utils import InputData, InputTypes, utils, mathUtils
from ...SupervisedLearning import factory as romFactory
from ... import GridEntities
from ... import Files
#Internal Modules End--------------------------------------------------------------------------------

class LimitSurface(PostProcessorInterface):
  """
    LimitSurface filter class. It computes the limit surface associated to a dataset
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
    inputSpecification = super().getInputSpecification()

    ParametersInput = InputData.parameterInputFactory("parameters", contentType=InputTypes.StringType)
    inputSpecification.addSub(ParametersInput)

    ToleranceInput = InputData.parameterInputFactory("tolerance", contentType=InputTypes.FloatType)
    inputSpecification.addSub(ToleranceInput)

    SideInput = InputData.parameterInputFactory("side", contentType=InputTypes.StringType)
    inputSpecification.addSub(SideInput)

    ROMInput = InputData.parameterInputFactory("ROM", contentType=InputTypes.StringType)
    ROMInput.addParam("class", InputTypes.StringType)
    ROMInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(ROMInput)

    FunctionInput = InputData.parameterInputFactory("Function", contentType=InputTypes.StringType)
    FunctionInput.addParam("class", InputTypes.StringType)
    FunctionInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(FunctionInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.parameters        = {}               #parameters dictionary (they are basically stored into a dictionary identified by tag "targets"
    self.surfPoint         = None             #coordinate of the points considered on the limit surface
    self.testMatrix        = OrderedDict()    #This is the n-dimensional matrix representing the testing grid
    self.gridCoord         = {}               #Grid coordinates
    self.functionValue     = {}               #This a dictionary that contains np vectors with the value for each variable and for the goal function
    self.ROM               = None             #Pointer to a ROM
    self.externalFunction  = None             #Pointer to an external Function
    self.tolerance         = 1.0e-4           #SubGrid tolerance
    self.gridFromOutside   = False            #The grid has been passed from outside (self._initFromDict)?
    self.lsSide            = "negative"       # Limit surface side to compute the LS for (negative,positive,both)
    self.gridEntity        = None             # the Grid object
    self.bounds            = None             # the container for the domain of search
    self.jobHandler        = None             # job handler pointer
    self.transfMethods     = {}               # transformation methods container
    self.crossedLimitSurf  = False            # Limit surface has been crossed?
    self.addAssemblerObject('ROM', InputData.Quantity.zero_to_one)
    self.addAssemblerObject('Function', InputData.Quantity.one)
    self.printTag = 'POSTPROCESSOR LIMITSURFACE'

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the samplers that need to request special objects
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    needDict = {'internal':[(None,'jobHandler')]}
    return needDict

  def _localGenerateAssembler(self,initDict):
    """
      Generates the assembler.
      @ In, initDict, dict, dict of init objects
      @ Out, None
    """
    self.jobHandler = initDict['internal']['jobHandler']

  def _initializeLSpp(self, runInfo, inputs, initDict):
    """
      Method to initialize the LS post processor (create grid, etc.)
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    self.gridEntity = GridEntities.factory.returnInstance("MultiGridEntity")
    self.externalFunction = self.assemblerDict['Function'][0][3]
    if 'ROM' not in self.assemblerDict.keys():
      self.ROM = romFactory.returnInstance('KNeighborsClassifier')
      paramDict = {'Features':list(self.parameters['targets']), 'Target':[self.externalFunction.name]}
      self.ROM.initializeFromDict(paramDict)
      settings = {"n_neighbors":1}
      self.ROM.initializeModel(settings)
    else:
      self.ROM = self.assemblerDict['ROM'][0][3]
    self.indexes = -1
    for index, inp in enumerate(self.inputs):
      if mathUtils.isAString(inp)  or isinstance(inp, bytes):
        self.raiseAnError(IOError, 'LimitSurface PostProcessor only accepts Data(s) as inputs. Got string type!')
      if inp.type == 'PointSet':
        self.indexes = index
    if self.indexes == -1:
      self.raiseAnError(IOError, 'LimitSurface PostProcessor needs a PointSet as INPUT!!!!!!')
    #else:
    #  # check if parameters are contained in the data
    #  inpKeys = self.inputs[self.indexes].getParaKeys("inputs")
    #  outKeys = self.inputs[self.indexes].getParaKeys("outputs")
    #  self.paramType = {}
    #  for param in self.parameters['targets']:
    #    if param not in inpKeys + outKeys:
    #      self.raiseAnError(IOError, 'LimitSurface PostProcessor: The param ' + param + ' not contained in Data ' + self.inputs[self.indexes].name + ' !')
    #    if param in inpKeys:
    #      self.paramType[param] = 'inputs'
    #    else:
    #      self.paramType[param] = 'outputs'
    if self.bounds == None:
      dataSet = self.inputs[self.indexes].asDataset()
      self.bounds = {"lowerBounds":{},"upperBounds":{}}
      for key in self.parameters['targets']:
        self.bounds["lowerBounds"][key], self.bounds["upperBounds"][key] = min(dataSet[key].values), max(dataSet[key].values)
        #self.bounds["lowerBounds"][key], self.bounds["upperBounds"][key] = min(self.inputs[self.indexes].getParam(self.paramType[key],key,nodeId = 'RecontructEnding')), max(self.inputs[self.indexes].getParam(self.paramType[key],key,nodeId = 'RecontructEnding'))
        if utils.compare(round(self.bounds["lowerBounds"][key],14),round(self.bounds["upperBounds"][key],14)):
          self.bounds["upperBounds"][key]+= abs(self.bounds["upperBounds"][key]/1.e7)
    self.gridEntity.initialize(initDictionary={"rootName":self.name,'constructTensor':True, "computeCells":initDict['computeCells'] if 'computeCells' in initDict.keys() else False,
                                               "dimensionNames":self.parameters['targets'], "lowerBounds":self.bounds["lowerBounds"],"upperBounds":self.bounds["upperBounds"],
                                               "volumetricRatio":self.tolerance   ,"transformationMethods":self.transfMethods})
    self.nVar                  = len(self.parameters['targets'])                                  # Total number of variables
    self.axisName              = self.gridEntity.returnParameter("dimensionNames",self.name)      # this list is the implicit mapping of the name of the variable with the grid axis ordering self.axisName[i] = name i-th coordinate
    self.testMatrix[self.name] = np.zeros(self.gridEntity.returnParameter("gridShape",self.name)) # grid where the values of the goalfunction are stored

  def _initializeLSppROM(self, inp, raiseErrorIfNotFound = True):
    """
      Method to initialize the LS acceleration rom
      @ In, inp, Data(s) object, data object containing the training set
      @ In, raiseErrorIfNotFound, bool, throw an error if the limit surface is not found
      @ Out, None
    """
    self.raiseADebug('Initiate training')
    if type(inp) == dict:
      self.functionValue.update(inp)
    else:
      dataSet = inp.asDataset("dict")
      self.functionValue.update(dataSet['data'])
      #self.functionValue.update(inp.getParametersValues('outputs', nodeId = 'RecontructEnding'))
    # recovery the index of the last function evaluation performed
    if self.externalFunction.name in self.functionValue.keys():
      indexLast = len(self.functionValue[self.externalFunction.name]) - 1
    else:
      indexLast = -1
    # index of last set of point tested and ready to perform the function evaluation
    indexEnd = len(self.functionValue[self.axisName[0]]) - 1
    tempDict = {}
    if self.externalFunction.name in self.functionValue.keys():
      self.functionValue[self.externalFunction.name] = np.append(self.functionValue[self.externalFunction.name], np.zeros(indexEnd - indexLast))
    else:
      self.functionValue[self.externalFunction.name] = np.zeros(indexEnd + 1)

    for myIndex in range(indexLast + 1, indexEnd + 1):
      for key, value in self.functionValue.items():
        tempDict[key] = value[myIndex]
      self.functionValue[self.externalFunction.name][myIndex] = self.externalFunction.evaluate('residuumSign', tempDict)
      if abs(self.functionValue[self.externalFunction.name][myIndex]) != 1.0:
        self.raiseAnError(IOError, 'LimitSurface: the function evaluation of the residuumSign method needs to return a 1 or -1!')
      if type(inp).__name__ in ['dict','OrderedDict']:
        if self.externalFunction.name in inp:
          inp[self.externalFunction.name] = np.concatenate((inp[self.externalFunction.name],np.asarray(self.functionValue[self.externalFunction.name][myIndex])))
    # check if the Limit Surface has been crossed
    self.crossedLimitSurf = not (np.sum(self.functionValue[self.externalFunction.name]) ==
                                 float(len(self.functionValue[self.externalFunction.name])) or
                                 np.sum(self.functionValue[self.externalFunction.name]) ==
                                 -float(len(self.functionValue[self.externalFunction.name])))
    if not self.crossedLimitSurf:
      if raiseErrorIfNotFound:
        self.raiseAnError(ValueError, 'LimitSurface: all the Function evaluations brought to the same result (No Limit Surface has been crossed...). Increase or change the data set!')
      else:
        self.raiseAWarning('LimitSurface: all the Function evaluations brought to the same result (No Limit Surface has been crossed...)!')
    #printing----------------------
    self.raiseADebug('LimitSurface: Mapping of the goal function evaluation performed')
    self.raiseADebug('LimitSurface: Already evaluated points and function values:')
    keyList = list(self.functionValue.keys())
    self.raiseADebug(','.join(keyList))
    for index in range(indexEnd + 1):
      self.raiseADebug(','.join([str(self.functionValue[key][index]) for key in keyList]))
    #printing----------------------
    tempDict = {}
    for name in self.axisName:
      tempDict[name] = np.asarray(self.functionValue[name])
    tempDict[self.externalFunction.name] = self.functionValue[self.externalFunction.name]
    self.ROM.train(tempDict)
    self.raiseADebug('LimitSurface: Training performed')

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the LS pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    self._initializeLSpp(runInfo, inputs, initDict)
    self._initializeLSppROM(self.inputs[self.indexes])

  def _initFromDict(self, dictIn):
    """
      Initialize the LS pp from a dictionary (not from xml input).
      This is used when other objects initialize and use the LS pp for internal
      calculations
      @ In, dictIn, dict, dictionary of initialization options
      @ Out, None
    """
    if "parameters" not in dictIn.keys():
      self.raiseAnError(IOError, 'No Parameters specified in "dictIn" dictionary !!!!')
    if "name" in dictIn.keys():
      self.name = dictIn["name"]
    if type(dictIn["parameters"]).__name__ == "list":
      self.parameters['targets'] = dictIn["parameters"]
    else:
      self.parameters['targets'] = dictIn["parameters"].split(",")
    if "bounds" in dictIn.keys():
      self.bounds = dictIn["bounds"]
    if "transformationMethods" in dictIn.keys():
      self.transfMethods = dictIn["transformationMethods"]
    if "verbosity" in dictIn.keys():
      self.verbosity = dictIn['verbosity']
    if "side" in dictIn.keys():
      self.lsSide = dictIn["side"]
    if "tolerance" in dictIn.keys():
      self.tolerance = float(dictIn["tolerance"])
    if self.lsSide not in ["negative", "positive", "both"]:
      self.raiseAnError(IOError, 'Computation side can be positive, negative, both only !!!!')

  def getFunctionValue(self):
    """
    Method to get a pointer to the dictionary self.functionValue
    @ In, None
    @ Out, dictionary, self.functionValue
    """
    return self.functionValue

  def getTestMatrix(self, nodeName=None,exceptionGrid=None):
    """
      Method to get a pointer to the testMatrix object (evaluation grid)
      @ In, nodeName, string, optional, which grid node should be returned. If None, the self.name one, If "all", all of theme, else the nodeName
      @ In, exceptionGrid, string, optional, which grid node should should not returned in case nodeName is "all"
      @ Out, testMatrix, numpy.ndarray or dict , self.testMatrix
    """
    if nodeName == None:
      testMatrix = self.testMatrix[self.name]
    elif nodeName =="all":
      if exceptionGrid == None:
        testMatrix = self.testMatrix
      else:
        returnDict = OrderedDict()
        wantedKeys = list(self.testMatrix.keys())
        wantedKeys.pop(wantedKeys.index(exceptionGrid))
        for key in wantedKeys:
          returnDict[key] = self.testMatrix[key]
        testMatrix = returnDict
    else:
      testMatrix = self.testMatrix[nodeName]
    return testMatrix

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    initDict = {}
    for child in paramInput.subparts:
      initDict[child.getName()] = child.value
    initDict.update(paramInput.parameterValues)
    self._initFromDict(initDict)

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    self.raiseADebug(str(evaluation))
    limitSurf = evaluation[1]
    if limitSurf[0] is not None:
      # reset the output
      if len(output) > 0:
        self.raiseAnError(RuntimeError, 'The output DataObject "'+output.name+'" is not empty! Chose another one!')
        #output.reset()
      # construct the realizations dict
      rlz = {varName: limitSurf[0][:,varIndex] for varIndex,varName in enumerate(self.axisName) }
      rlz[self.externalFunction.name] = limitSurf[1]
      # add the full realizations
      output.load(rlz,style='dict')

  def refineGrid(self,refinementSteps=2):
    """
      Method to refine the internal grid based on the limit surface previously computed
      @ In, refinementSteps, int, optional, number of refinement steps
      @ Out, None
    """
    cellIds = self.gridEntity.retrieveCellIds([self.listSurfPointNegative,self.listSurfPointPositive],self.name)
    if self.getVerbosity() == 'debug':
      self.raiseADebug("Limit Surface cell IDs are: \n"+ " \n".join([str(cellID) for cellID in cellIds]))
    self.raiseAMessage("Number of cells to be refined are "+str(len(cellIds))+". RefinementSteps = "+str(max([refinementSteps,2]))+"!")
    self.gridEntity.refineGrid({"cellIDs":cellIds,"refiningNumSteps":int(max([refinementSteps,2]))})
    for nodeName in self.gridEntity.getAllNodesNames(self.name):
      if nodeName != self.name:
        self.testMatrix[nodeName] = np.zeros(self.gridEntity.returnParameter("gridShape",nodeName))

  def run(self, inputIn = None, returnListSurfCoord = False, exceptionGrid = None, merge = True):
    """
      This method executes the postprocessor action. In this case it computes the limit surface.
      @ In, inputIn, dict, optional, dictionary of data to process
      @ In, returnListSurfCoord, bool, optional, True if listSurfaceCoordinate needs to be returned
      @ In, exceptionGrid, string, optional, the name of the sub-grid to not be considered
      @ In, merge, bool, optional, True if the LS in all the sub-grids need to be merged in a single returnSurface
      @ Out, returnSurface, tuple, tuple containing the limit surface info:
                          - if returnListSurfCoord: returnSurface = (surfPoint, evals, listSurfPoints)
                          - else                  : returnSurface = (surfPoint, evals)
    """
    allGridNames = self.gridEntity.getAllNodesNames(self.name)
    if exceptionGrid != None:
      try:
        allGridNames.pop(allGridNames.index(exceptionGrid))
      except:
        pass
    self.surfPoint, evaluations, listSurfPoint = OrderedDict().fromkeys(allGridNames), OrderedDict().fromkeys(allGridNames) ,OrderedDict().fromkeys(allGridNames)
    for nodeName in allGridNames:
      #if skipMainGrid == True and nodeName == self.name: continue
      self.testMatrix[nodeName] = np.zeros(self.gridEntity.returnParameter("gridShape",nodeName))
      self.gridCoord[nodeName] = self.gridEntity.returnGridAsArrayOfCoordinates(nodeName=nodeName)
      tempDict ={}
      for  varId, varName in enumerate(self.axisName):
        tempDict[varName] = self.gridCoord[nodeName][:,varId]
      self.testMatrix[nodeName].shape     = (self.gridCoord[nodeName].shape[0])                       #rearrange the grid matrix such as is an array of values
      self.testMatrix[nodeName][:]        = self.ROM.evaluate(tempDict)[self.externalFunction.name]   #get the prediction on the testing grid
      self.testMatrix[nodeName].shape     = self.gridEntity.returnParameter("gridShape",nodeName)     #bring back the grid structure
      self.gridCoord[nodeName].shape      = self.gridEntity.returnParameter("gridCoorShape",nodeName) #bring back the grid structure
      self.raiseADebug('LimitSurface: Prediction performed')
      # here next the points that are close to any change are detected by a gradient (it is a pre-screener)
      if self.nVar > 1:
        toBeTested = np.squeeze(np.dstack(np.nonzero(np.sum(np.abs(np.gradient(self.testMatrix[nodeName])), axis = 0))))
      else:
        toBeTested = np.squeeze(np.dstack(np.nonzero(np.abs(np.gradient(self.testMatrix[nodeName])))))
      toBeTested = np.atleast_2d(toBeTested).T if self.nVar == 1 else toBeTested
      #printing----------------------
      self.raiseADebug('LimitSurface:  Limit surface candidate points')
      if self.getVerbosity() == 'debug':
        for coordinate in np.rollaxis(toBeTested, 0):
          myStr = ''
          for iVar, varnName in enumerate(self.axisName):
            myStr += varnName + ': ' + str(coordinate[iVar]) + '      '
          self.raiseADebug('LimitSurface: ' + myStr + '  value: ' + str(self.testMatrix[nodeName][tuple(coordinate)]))
      # printing----------------------
      # check which one of the preselected points is really on the limit surface
      nNegPoints, nPosPoints                       =  0, 0
      listSurfPointNegative, listSurfPointPositive = [], []
      if self.lsSide in ["negative", "both"]:
        # it returns the list of points belonging to the limit state surface and resulting in a negative response by the ROM
        listSurfPointNegative = self.__localLimitStateSearch__(toBeTested, -1, nodeName)
        nNegPoints = len(listSurfPointNegative)
      if self.lsSide in ["positive", "both"]:
        # it returns the list of points belonging to the limit state surface and resulting in a positive response by the ROM
        listSurfPointPositive = self.__localLimitStateSearch__(toBeTested, 1, nodeName)
        nPosPoints = len(listSurfPointPositive)
      listSurfPoint[nodeName] = listSurfPointNegative + listSurfPointPositive
      #printing----------------------
      if self.getVerbosity() == 'debug':
        if len(listSurfPoint[nodeName]) > 0:
          self.raiseADebug('LimitSurface: Limit surface points:')
        for coordinate in listSurfPoint[nodeName]:
          myStr = ''
          for iVar, varnName in enumerate(self.axisName):
            myStr += varnName + ': ' + str(coordinate[iVar]) + '      '
          self.raiseADebug('LimitSurface: ' + myStr + '  value: ' + str(self.testMatrix[nodeName][tuple(coordinate)]))
      # if the number of point on the limit surface is > than zero than save it
      if len(listSurfPoint[nodeName]) > 0:
        self.surfPoint[nodeName] = np.ndarray((len(listSurfPoint[nodeName]), self.nVar))
        evaluations[nodeName] = np.concatenate((-np.ones(nNegPoints), np.ones(nPosPoints)), axis = 0)
        for pointID, coordinate in enumerate(listSurfPoint[nodeName]):
          self.surfPoint[nodeName][pointID, :] = self.gridCoord[nodeName][tuple(coordinate)]
    if self.name != exceptionGrid:
      self.listSurfPointNegative, self.listSurfPointPositive = listSurfPoint[self.name][:nNegPoints-1],listSurfPoint[self.name][nNegPoints:]
    if merge == True:
      evals = np.hstack(evaluations.values())
      listSurfPoints = np.hstack(listSurfPoint.values())
      surfPoint = np.hstack(self.surfPoint.values())
      returnSurface = (surfPoint, evals, listSurfPoints) if returnListSurfCoord else (surfPoint, evals)
    else:
      returnSurface = (self.surfPoint, evaluations, listSurfPoint) if returnListSurfCoord else (self.surfPoint, evaluations)
    return returnSurface

  def __localLimitStateSearch__(self, toBeTested, sign, nodeName):
    """
      It returns the list of points belonging to the limit state surface and resulting in
      positive or negative responses by the ROM, depending on whether ''sign''
      equals either -1 or 1, respectively.
      @ In, toBeTested, np.ndarray, the nodes to be tested
      @ In, sign, int, the sign that should be tested (-1 or +1)
      @ In, nodeName, string, the sub-grid name
      @ Out, listSurfPoint, list, the list of limit surface coordinates
    """
    #TODO: This approach might be extremely slow. It must be replaced with matrix operations
    listSurfPoint = []
    gridShape = self.gridEntity.returnParameter("gridShape",nodeName)
    myIdList = np.zeros(self.nVar,dtype=int)
    putIt = np.zeros(self.nVar,dtype=bool)
    for coordinate in np.rollaxis(toBeTested, 0):
      myIdList[:] = coordinate
      putIt[:]    = False
      if self.testMatrix[nodeName][tuple(coordinate)] * sign > 0:
        for iVar in range(self.nVar):
          if coordinate[iVar] + 1 < gridShape[iVar]:
            myIdList[iVar] += 1
            if self.testMatrix[nodeName][tuple(myIdList)] * sign <= 0:
              putIt[iVar] = True
              listSurfPoint.append(copy.copy(coordinate))
              break
            myIdList[iVar] -= 1
            if coordinate[iVar] > 0:
              myIdList[iVar] -= 1
              if self.testMatrix[nodeName][tuple(myIdList)] * sign <= 0:
                putIt[iVar] = True
                listSurfPoint.append(copy.copy(coordinate))
                break
              myIdList[iVar] += 1
      #if len(set(putIt)) == 1 and  list(set(putIt))[0] == True: listSurfPoint.append(copy.copy(coordinate))
    return listSurfPoint
