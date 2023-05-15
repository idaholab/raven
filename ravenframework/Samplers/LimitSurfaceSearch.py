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
  This module contains the Limit Surface Search sampling strategy

  Created on May 21, 2016
  @author: alfoa
  supercedes Samplers.py from alfoa
"""
from collections import OrderedDict
import copy
import numpy as np
from operator import mul
from functools import reduce
from scipy import spatial
from math import ceil

from .. import Distributions
from AMSC.AMSC_Object import AMSC_Object
from ..utils import randomUtils
from ..utils import InputData, InputTypes
from .AdaptiveSampler import AdaptiveSampler


class LimitSurfaceSearch(AdaptiveSampler):
  """
    A sampler that will adaptively locate the limit surface of a given problem
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
    inputSpecification = super(LimitSurfaceSearch, cls).getInputSpecification()

    convergenceInput = InputData.parameterInputFactory("Convergence", contentType=InputTypes.FloatType)
    convergenceInput.addParam("limit", InputTypes.IntegerType)
    convergenceInput.addParam("forceIteration", InputTypes.StringType)
    convergenceInput.addParam("weight", InputTypes.StringType)
    convergenceInput.addParam("persistence", InputTypes.IntegerType)
    convergenceInput.addParam("subGridTol", InputTypes.FloatType)

    inputSpecification.addSub(convergenceInput)

    batchStrategyInput = InputData.parameterInputFactory("batchStrategy",
                                                         contentType=InputTypes.StringType)
    inputSpecification.addSub(batchStrategyInput)

    maxBatchSizeInput = InputData.parameterInputFactory("maxBatchSize", contentType=InputTypes.IntegerType)
    inputSpecification.addSub(maxBatchSizeInput)
    scoringInput = InputData.parameterInputFactory("scoring", contentType=InputTypes.StringType)
    inputSpecification.addSub(scoringInput)
    simplificationInput = InputData.parameterInputFactory("simplification", contentType=InputTypes.FloatType)
    inputSpecification.addSub(simplificationInput)

    thicknessInput = InputData.parameterInputFactory("thickness", contentType=InputTypes.IntegerType)
    inputSpecification.addSub(thicknessInput)

    thresholdInput = InputData.parameterInputFactory("threshold", contentType=InputTypes.FloatType)
    inputSpecification.addSub(thresholdInput)

    romInput = InputData.parameterInputFactory("ROM", contentType=InputTypes.StringType)
    romInput.addParam("type", InputTypes.StringType)
    romInput.addParam("class", InputTypes.StringType)
    inputSpecification.addSub(romInput)

    targetEvaluationInput = InputData.parameterInputFactory("TargetEvaluation", contentType=InputTypes.StringType)
    targetEvaluationInput.addParam("type", InputTypes.StringType)
    targetEvaluationInput.addParam("class", InputTypes.StringType)
    inputSpecification.addSub(targetEvaluationInput)

    functionInput = InputData.parameterInputFactory("Function", contentType=InputTypes.StringType)
    functionInput.addParam("type", InputTypes.StringType)
    functionInput.addParam("class", InputTypes.StringType)
    inputSpecification.addSub(functionInput)

    return inputSpecification

  @classmethod
  def getSolutionExportVariableNames(cls):
    """
      Compiles a list of acceptable SolutionExport variable options.
      @ In, None
      @ Out, ok, dict, {varName: manual description} for each solution export option
    """
    # cannot be determined before run-time due to variables and prefixes.
    ok = super(LimitSurfaceSearch, cls).getSolutionExportVariableNames()
    new = {'{VAR}': 'Variable values from the TargetEvaluation DataObject',
           '{RESIDUUM}': 'RAVEN input name of module containing __residuumSign method; provides the evaluation of the function.'
          }
    ok.update(new)
    return ok

  def __init__(self):
    """
      Default Constructor that will initialize member variables with reasonable
      defaults or empty lists/dictionaries where applicable.
      @ In, None
      @ Out, None
    """
    AdaptiveSampler.__init__(self)
    self.goalFunction        = None             #this is the pointer to the function defining the goal
    self.tolerance           = None             #this is norm of the error threshold
    self.subGridTol          = None             #This is the tolerance used to construct the testing sub grid
    self.toleranceWeight     = 'cdf'            #this is the a flag that controls if the convergence is checked on the hyper-volume or the probability
    self.persistence         = 5                #this is the number of times the error needs to fell below the tolerance before considering the sim converged
    self.repetition          = 0                #the actual number of time the error was below the requested threshold
    self.forceIteration      = False            #this flag control if at least a self.limit number of iteration should be done
    self.axisName            = None             #this is the ordered list of the variable names (ordering match self.gridStepSize and the ordering in the test matrixes)
    self.oldTestMatrix       = OrderedDict()    #This is the test matrix to use to store the old evaluation of the function
    self.persistenceMatrix   = OrderedDict()    #this is a matrix that for each point of the testing grid tracks the persistence of the limit surface position
    self.invPointPersistence = OrderedDict()    #this is a matrix that for each point of the testing grid tracks the inverse of the persistence of the limit surface position
    self.solutionExport      = None             #This is the data used to export the solution (it could also not be present)
    self.nVar                = 0                #this is the number of the variable sampled
    self.surfPoint           = None             #coordinate of the points considered on the limit surface
    self.hangingPoints       = []               #list of the points already submitted for evaluation for which the result is not yet available
    self.refinedPerformed    = False            # has the grid refinement been performed?
    self.limitSurfacePP      = None             # post-processor to compute the limit surface
    self.exceptionGrid       = None             # which cell should be not considered in the limit surface computation? set by refinement
    self.errorTolerance      = 1.0              # initial error tolerance (number of points can change between iterations in LS search)
    self.jobHandler          = None             # jobHandler for generation of grid in parallel
    self.firstSurface        = True             # if first LS do not consider the invPointPersistence information (if true)
    self.scoringMethod  = 'distancePersistence' # The scoring method to use
    self.batchStrategy  = 'none'                # The batch strategy to use
    self.converged      = False                 # flag that is set to True when the sampler converged
    # self.generateCSVs   = False                 # Flag: should intermediate
    #                                             #  results be stored?
    self.toProcess      = []                    # List of the top batchSize
                                                #  candidates that will be
                                                #  populated and depopulated
                                                #  during subsequent calls of
                                                #  localGenerateInput
    self.maxBatchSize   = None                  # Maximum batch size, the top
                                                #  candidates will be selected,
                                                #  if there are more local
                                                #  maxima than this value, then
                                                #  we wiil only take the top
                                                #  persistence ones, if there
                                                #  are fewer, then we will only
                                                #  grab that many and then force
                                                #  an early update
    self.thickness      = 0                      # Number of steps outward from
                                                #  the extracted limit surface
                                                #  to include in the candidate
                                                #  set
    self.simplification = 0                     # Pre-rank simpligication level
                                                #  (% of range space)
    self.threshold      = 0                     # Post-rank function value
                                                #  cutoff (%  of range space)
    self.sizeGrid       = None                  # size of grid
    self.sizeSubGrid    = None                  # size of subgrid
    self.printTag            = 'SAMPLER ADAPTIVE'

    self.acceptedScoringParam = ['distance','distancePersistence']
    self.acceptedBatchParam = ['none','naive','maxV','maxP']

    self.addAssemblerObject('ROM', InputData.Quantity.one_to_infinity)
    self.addAssemblerObject('Function', InputData.Quantity.zero_to_infinity)

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the samplers that need to request special objects
      @ In, None
      @ Out, limitSurfaceDict, dict, list of objects needed
    """
    limitSurfaceDict = AdaptiveSampler._localWhatDoINeed(self)
    limitSurfaceDict['internal'] = [(None,'jobHandler')]
    return limitSurfaceDict

  def _localGenerateAssembler(self,initDict):
    """
      Generates the assembler.
      @ In, initDict, dict, dict of init objects
      @ Out, None
    """
    AdaptiveSampler._localGenerateAssembler(self, initDict)
    self.jobHandler = initDict['internal']['jobHandler']
    #do a distributions check for ND
    for dist in self.distDict.values():
      if isinstance(dist,Distributions.NDimensionalDistributions):
        self.raiseAnError(IOError,'ND Dists not supported for this sampler (yet)!')

  def localInputAndChecks(self,xmlNode, paramInput):
    """
      Class specific xml inputs will be read here and checked for validity.
      @ In, xmlNode, xml.etree.ElementTree.Element, The xml element node that will be checked against the available options specific to this Sampler.
      @ In, paramInput, InputData.ParameterInput, the parsed parameters
      @ Out, None
    """
    # check input variables to avoid <variable name='x1,x2,...'> node
    for val in list(self.toBeSampled.keys()) + list(self.dependentSample.keys()):
      if len(val.split(',')) > 1:
        self.raiseAnError(IOError, f'Variables {val} defined using <variable> node can not be accepted, only single variable can be processed!')
    #TODO remove using xmlNode
    if 'limit' in xmlNode.attrib.keys():
      try:
        self.limit = int(xmlNode.attrib['limit'])
      except ValueError:
        self.raiseAnError(IOError,'reading the attribute for the sampler '+self.name+' it was not possible to perform the conversion to integer for the attribute limit with value '+xmlNode.attrib['limit'])
    # convergence Node
    convergenceNode = xmlNode.find('Convergence')
    if convergenceNode==None:
      self.raiseAnError(IOError,'the node Convergence was missed in the definition of the adaptive sampler '+self.name)
    try:
      self.tolerance=float(convergenceNode.text)
    except:
      self.raiseAnError(IOError,'Failed to convert '+convergenceNode.text+' to a meaningful number for the convergence')
    self.errorTolerance = self.tolerance
    attribList = list(convergenceNode.attrib.keys())
    if 'limit'          in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('limit'))
      try:
        self.limit = int (convergenceNode.attrib['limit'])
      except:
        self.raiseAnError(IOError,'Failed to convert the limit value '+convergenceNode.attrib['limit']+' to a meaningful number for the convergence')
    if 'persistence'    in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('persistence'))
      try:
        self.persistence = int (convergenceNode.attrib['persistence'])
      except:
        self.raiseAnError(IOError,'Failed to convert the persistence value '+convergenceNode.attrib['persistence']+' to a meaningful number for the convergence')
    if 'weight'         in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('weight'))
      try:
        self.toleranceWeight = str(convergenceNode.attrib['weight']).lower()
      except:
        self.raiseAnError(IOError,'Failed to convert the weight type '+convergenceNode.attrib['weight']+' to a meaningful string for the convergence')
    if 'subGridTol'    in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('subGridTol'))
      try:
        self.subGridTol = float (convergenceNode.attrib['subGridTol'])
      except:
        self.raiseAnError(IOError,'Failed to convert the subGridTol '+convergenceNode.attrib['subGridTol']+' to a meaningful float for the convergence')
    if 'forceIteration' in convergenceNode.attrib.keys():
      attribList.pop(attribList.index('forceIteration'))
      if   convergenceNode.attrib['forceIteration']=='True':
        self.forceIteration   = True
      elif convergenceNode.attrib['forceIteration']=='False':
        self.forceIteration   = False
      else:
        self.raiseAnError(RuntimeError,'Reading the convergence setting for the adaptive sampler '+self.name+' the forceIteration keyword had an unknown value: '+str(convergenceNode.attrib['forceIteration']))
    #assembler node: Hidden from User
    # set subgrid
    if self.subGridTol == None:
      self.subGridTol = self.tolerance
    if self.subGridTol > self.tolerance:
      self.raiseAnError(IOError,'The sub grid tolerance '+str(self.subGridTol)+' must be smaller than the tolerance: '+str(self.tolerance))
    if len(attribList)>0:
      self.raiseAnError(IOError,'There are unknown keywords in the convergence specifications: '+str(attribList))

    # Batch parameters
    for child in xmlNode:
      if child.tag == "generateCSVs":
        self.generateCSVs = True
      if child.tag == "batchStrategy":
        self.batchStrategy = child.text
        if self.batchStrategy not in self.acceptedBatchParam:
          self.raiseAnError(IOError, 'Requested unknown batch strategy: ',
                            self.batchStrategy, '. Available options: ',
                            self.acceptedBatchParam)
      if child.tag == "maxBatchSize":
        try:
          self.maxBatchSize = int(child.text)
        except:
          self.raiseAnError(IOError, 'Failed to convert the maxBatchSize value: ' + child.text + ' into a meaningful integer')
        if self.maxBatchSize < 0:
          self.raiseAWarning(IOError,'Requested an invalid maximum batch size: ', self.maxBatchSize, '. This should be a non-negative integer value. Defaulting to 1.')
          self.maxBatchSize = 1
      if child.tag == "scoring":
        self.scoringMethod = child.text
        if self.scoringMethod not in self.acceptedScoringParam:
          self.raiseAnError(IOError, 'Requested unknown scoring type: ', self.scoringMethod, '. Available options: ', self.acceptedScoringParam)
      if child.tag == 'simplification':
        try:
          self.simplification = float(child.text)
        except:
          self.raiseAnError(IOError, 'Failed to convert the simplification value: ' + child.text + ' into a meaningful number')
        if self.simplification < 0 or self.simplification > 1:
          self.raiseAWarning('Requested an invalid simplification level: ', self.simplification, '. Defaulting to 0.')
          self.simplification = 0
      if child.tag == 'thickness':
        try:
          self.thickness = int(child.text)
        except:
          self.raiseAnError(IOError, 'Failed to convert the thickness value: ' + child.text +' into a meaningful integer')
        if self.thickness < 0:
          self.raiseAWarning('Requested an invalid thickness size: ', self.thickness, '. Defaulting to 0.')
      if child.tag == 'threshold':
        try:
          self.threshold = float(child.text)
        except:
          self.raiseAnError(IOError, 'Failed to convert the threshold value: ' + child.text + ' into a meaningful number')
        if self.threshold < 0 or self.threshold > 1:
          self.raiseAWarning('Requested an invalid threshold level: ', self.threshold, '. Defaulting to 0.')
          self.threshold = 0

  def localGetInitParams(self):
    """
      Appends a given dictionary with class specific member variables and their
      associated initialized values.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    paramDict['Iter. forced'    ] = str(self.forceIteration)
    paramDict['Norm tolerance'  ] = str(self.tolerance)
    paramDict['Sub grid size'   ] = str(self.subGridTol)
    paramDict['Error Weight'    ] = str(self.toleranceWeight)
    paramDict['Persistence'     ] = str(self.repetition)
    paramDict['batchStrategy'   ] = self.batchStrategy
    paramDict['maxBatchSize'    ] = self.maxBatchSize
    paramDict['scoring'         ] = str(self.scoringMethod)
    paramDict['simplification'  ] = self.simplification
    paramDict['thickness'       ] = self.thickness
    paramDict['threshold'       ] = self.threshold
    return paramDict

  def localGetCurrentSetting(self):
    """
      Appends a given dictionary with class specific information regarding the
      current status of the object.
      @ In, None
      @ Out, paramDict, dict, dictionary containing the parameter names as keys
        and each parameter's initial value as the dictionary values
    """
    paramDict = {}
    if self.solutionExport!=None:
      paramDict['The solution is exported in '    ] = 'Name: ' + self.solutionExport.name + 'Type: ' + self.solutionExport.type
    if self.goalFunction!=None  :
      paramDict['The function used is '] = self.goalFunction.name
    return paramDict

  def localInitialize(self,solutionExport=None):
    """
      Will perform all initialization specific to this Sampler. For instance,
      creating an empty container to hold the identified surface points, error
      checking the optionally provided solution export and other preset values,
      and initializing the limit surface Post-Processor used by this sampler.
      @ In, solutionExport, DataObjects, optional, a PointSet to hold the solution (a list of limit surface points)
      @ Out, None
    """
    self.converged = False
    from ..Models.PostProcessors import factory as ppFactory
    self.limitSurfacePP = ppFactory.returnInstance('LimitSurface')
    if 'Function' in self.assemblerDict.keys():
      self.goalFunction = self.assemblerDict['Function'][0][3]
    # if 'TargetEvaluation' in self.assemblerDict.keys():
    self.lastOutput = self._targetEvaluation #self.assemblerDict['TargetEvaluation'][0][3]
    #self.memoryStep        = 5               # number of step for which the memory is kept
    self.solutionExport    = solutionExport
    # check if solutionExport is actually a "DataObjects" type "PointSet"
    if solutionExport.type != "PointSet":
      self.raiseAnError(IOError,'solutionExport type is not a PointSet. Got '+ solutionExport.type +'!')
    self.surfPoint         = None             #coordinate of the points considered on the limit surface
    self.oldTestMatrix     = OrderedDict()    #This is the test matrix to use to store the old evaluation of the function
    self.persistenceMatrix = OrderedDict()    #this is a matrix that for each point of the testing grid tracks the persistence of the limit surface position
    if self.goalFunction.name not in self.solutionExport.getVars('output'):
      self.raiseAnError(IOError,'Goal function name does not match solution export data output.')
    # set number of job request-able after a new evaluation
    self._endJobRunnable   = 1
    #check if convergence is not on probability if all variables are bounded in value otherwise the problem is unbounded
    if self.toleranceWeight=='value':
      for varName in self.distDict.keys():
        if not(self.distDict[varName].upperBoundUsed and self.distDict[varName].lowerBoundUsed):
          self.raiseAnError(TypeError,'It is impossible to converge on an unbounded domain (variable '+varName+' with distribution '+self.distDict[varName].name+') as requested to the sampler '+self.name)
    elif self.toleranceWeight=='cdf':
      pass
    else:
      self.raiseAnError(IOError,'Unknown weight string descriptor: '+self.toleranceWeight)
    #setup the grid. The grid is build such as each element has a volume equal to the sub grid tolerance
    #the grid is build in such a way that an unit change in each node within the grid correspond to a change equal to the tolerance
    self.nVar         = len(self.distDict.keys())              # Total number of variables
    bounds          = {"lowerBounds":{},"upperBounds":{}}
    transformMethod = {}
    for varName in self.distDict.keys():
      if self.toleranceWeight!='cdf':
        bounds["lowerBounds"][varName.replace('<distribution>','')], bounds["upperBounds"][varName.replace('<distribution>','')] = self.distDict[varName].lowerBound, self.distDict[varName].upperBound
      else:
        bounds["lowerBounds"][varName.replace('<distribution>','')], bounds["upperBounds"][varName.replace('<distribution>','')] = 0.0, 1.0
        transformMethod[varName.replace('<distribution>','')] = [self.distDict[varName].ppf]
    #moving forward building all the information set
    self.axisName = list(self.distDict.keys())
    self.axisName.sort()
    # initialize LimitSurface PP
    self.limitSurfacePP._initFromDict({"name":self.name+"LSpp","parameters":[key.replace('<distribution>','') for key in self.axisName],"tolerance":self.tolerance,"side":"both","transformationMethods":transformMethod,"bounds":bounds})
    self.limitSurfacePP.assemblerDict = self.assemblerDict
    self.limitSurfacePP._initializeLSpp({'WorkingDir': None},
                                        [self.lastOutput],
                                        {'computeCells':self.tolerance != self.subGridTol})
    matrixShape = self.limitSurfacePP.getTestMatrix().shape
    self.persistenceMatrix[self.name+"LSpp"]  = np.zeros(matrixShape) #matrix that for each point of the testing grid tracks the persistence of the limit surface position
    self.oldTestMatrix[self.name+"LSpp"]      = np.zeros(matrixShape) #swap matrix fro convergence test
    self.hangingPoints                        = np.ndarray((0, self.nVar))
    self.raiseADebug('Initialization done')

  def localStillReady(self,ready):
    """
      first perform some check to understand what it needs to be done possibly perform an early return
      ready is returned
      ROM if passed in it is used to construct the test matrix otherwise the nearest neighbor value is used
      @ In,  ready, bool, a boolean representing whether the caller is prepared for another input.
      @ Out, ready, bool, a boolean representing whether the caller is prepared for another input.
    """
    self.raiseADebug('From method localStillReady...')
    # if the limit surface search has converged, we return False
    if self.converged:
      return False
    #test on what to do
    if not ready:
      return ready #if we exceeded the limit just return that we are done
    if type(self.lastOutput) == dict:
      if self.lastOutput == None and not self.limitSurfacePP.ROM.amITrained:
        return ready
    else:
      #if the last output is not provided I am still generating an input batch, if the rom was not trained before we need to start clean
      if len(self.lastOutput) == 0 and not self.limitSurfacePP.ROM.amITrained:
        return ready
    #first evaluate the goal function on the newly sampled points and store them in mapping description self.functionValue RecontructEnding
    oldSizeLsFunctionValue = 0 if len(self.limitSurfacePP.getFunctionValue()) == 0 else len(self.limitSurfacePP.getFunctionValue()[self.goalFunction.name])
    if type(self.lastOutput) == dict:
      self.limitSurfacePP._initializeLSppROM(self.lastOutput,False)
    else:
      if len(self.lastOutput) > 0:
        self.limitSurfacePP._initializeLSppROM(self.lastOutput,False)
    self.raiseADebug('Classifier ' +self.name+' has been trained!')
    self.oldTestMatrix = copy.deepcopy(self.limitSurfacePP.getTestMatrix("all",exceptionGrid=self.exceptionGrid))    #copy the old solution (contained in the limit surface PP) for convergence check
    # evaluate the Limit Surface coordinates (return input space coordinates, evaluation vector and grid indexing)
    self.surfPoint, evaluations, self.listSurfPoint = self.limitSurfacePP.run(returnListSurfCoord = True, exceptionGrid=self.exceptionGrid, merge=False)
    self.raiseADebug('Limit Surface has been computed!')
    newSizeLsFunctionValue = len(self.limitSurfacePP.getFunctionValue()[self.goalFunction.name])  if self.goalFunction.name in self.limitSurfacePP.getFunctionValue().keys() else 0
    # check hanging points
    if self.goalFunction.name in self.limitSurfacePP.getFunctionValue().keys():
      indexLast = len(self.limitSurfacePP.getFunctionValue()[self.goalFunction.name])-1
    else:
      indexLast = -1
    #index of last set of point tested and ready to perform the function evaluation
    indexEnd  = len(self.limitSurfacePP.getFunctionValue()[self.axisName[0].replace('<distribution>','')])-1
    tempDict  = {}
    for myIndex in range(indexLast+1,indexEnd+1):
      for key, value in self.limitSurfacePP.getFunctionValue().items():
        tempDict[key] = value[myIndex]
      if len(self.hangingPoints) > 0:
        self.hangingPoints = self.hangingPoints[
          ~(self.hangingPoints==np.array([tempDict[varName]
                                          for varName in [key.replace('<distribution>','')
                                                          for key in self.axisName]])).all(axis=1)][:]
    for key,value in self.limitSurfacePP.getTestMatrix("all",exceptionGrid=self.exceptionGrid).items():
      self.persistenceMatrix[key] += value

    # get the test matrices' dictionaries to test the error
    testMatrixDict = list(self.limitSurfacePP.getTestMatrix("all",exceptionGrid=self.exceptionGrid).values())
    oldTestMatrixDict = list(self.oldTestMatrix.values())
    # the first test matrices in the list are always represented by the coarse grid
    # (if subGridTol activated) or the only grid available
    coarseGridTestMatix, coarseGridOldTestMatix = testMatrixDict.pop(0), oldTestMatrixDict.pop(0)
    # compute the Linf norm with respect the location of the LS
    testError = np.sum(np.abs(np.subtract(coarseGridTestMatix,coarseGridOldTestMatix)))
    if self.sizeGrid is None:
      self.sizeGrid = float(coarseGridTestMatix.size)
    if len(testMatrixDict) > 0:
      # compute the error
      if self.sizeSubGrid is None:
        self.sizeSubGrid = float(np.asarray(testMatrixDict).size)
      testError += np.sum(np.abs(np.subtract(testMatrixDict,oldTestMatrixDict)))/(self.sizeGrid+self.sizeSubGrid)
    else:
      testError/= self.sizeGrid

    if (testError > self.errorTolerance) or newSizeLsFunctionValue == oldSizeLsFunctionValue:
      # we still have error
      ready, self.repetition = True, 0
    else:
      # we are increasing persistence
      self.repetition +=1
    if self.persistence<self.repetition:
      ready =  False
      if self.subGridTol != self.tolerance \
         and evaluations is not None \
         and not self.refinedPerformed and self.limitSurfacePP.crossedLimitSurf:
        # we refine the grid since we converged on the coarse one. we use the "ceil" method in order to be sure
        # that the volumetric cell weight is <= of the subGridTol
        self.raiseAMessage("Grid refinement activated! Refining the evaluation grid!")
        self.limitSurfacePP.refineGrid(int(ceil((self.tolerance/self.subGridTol)**(1.0/self.nVar))))
        self.exceptionGrid, self.refinedPerformed, ready, self.repetition = self.name + "LSpp", True, True, 0
        self.persistenceMatrix.update(copy.deepcopy(self.limitSurfacePP.getTestMatrix("all",exceptionGrid=self.exceptionGrid)))
        self.errorTolerance = self.subGridTol
      else:
        self.converged = True
        if not self.limitSurfacePP.crossedLimitSurf:
          self.raiseAWarning("THE LIMIT SURFACE has NOT been crossed. The search FAILED!!!")
    self.raiseAMessage('counter: '+str(self.counter)+'       Error: {:9.6E} Repetition: {:5d}'.format(testError,self.repetition) )
    #if the number of point on the limit surface is > than compute persistence
    realAxisNames, cnt = [key.replace('<distribution>','') for key in self.axisName], 0
    if self.solutionExport is not None:
      rlz = {}
      # reset solution export
      self.solutionExport.reset()
    for gridID,listsurfPoint in self.listSurfPoint.items():
      if len(listsurfPoint)>0:
        self.invPointPersistence[gridID] = np.ones(len(listsurfPoint))
        if self.firstSurface == False:
          for pointID, coordinate in enumerate(listsurfPoint):
            self.invPointPersistence[gridID][pointID]=abs(self.persistenceMatrix[gridID][tuple(coordinate)])
          maxPers = np.max(self.invPointPersistence[gridID])
          if maxPers != 0:
            self.invPointPersistence[gridID] = (maxPers-self.invPointPersistence[gridID])/maxPers
        else:
          self.firstSurface = False
        if self.solutionExport is not None:
          # construct the realizations dict
          localRlz = {varName: (self.surfPoint[gridID][:,varIndex] if varName not in rlz else np.concatenate(( rlz[varName],self.surfPoint[gridID][:,varIndex] )) ) for varIndex,varName in enumerate(realAxisNames) }
          localRlz[self.goalFunction.name] = evaluations[gridID] if self.goalFunction.name not in rlz else np.concatenate( (rlz[self.goalFunction.name],evaluations[gridID])  )
          rlz.update(localRlz)
    # add the full realizations
    if self.solutionExport is not None:
      if len(rlz):
        self.solutionExport.load(rlz,style='dict')

    # Keep track of some extra points that we will add to thicken the limit
    # surface candidate set
    self.bandIndices = OrderedDict()
    for gridID,points in self.listSurfPoint.items():
      setSurfPoint = set()
      self.bandIndices[gridID] = set()
      for surfPoint in points:
        setSurfPoint.add(tuple(surfPoint))
      newIndices = set(setSurfPoint)
      for step in range(1,self.thickness):
        prevPoints = set(newIndices)
        newIndices = set()
        for i,iCoords in enumerate(prevPoints):
          for d in range(len(iCoords)):
            offset = np.zeros(len(iCoords),dtype=int)
            offset[d] = 1
            if iCoords[d] - offset[d] > 0:
              newIndices.add(tuple(iCoords - offset))
            if iCoords[d] + offset[d] < self.oldTestMatrix[gridID].shape[d]-1:
              newIndices.add(tuple(iCoords + offset))
        self.bandIndices[gridID].update(newIndices)
      self.bandIndices[gridID] = self.bandIndices[gridID].difference(setSurfPoint)
      self.bandIndices[gridID] = list(self.bandIndices[gridID])
      for coordinate in self.bandIndices[gridID]:
        self.surfPoint[gridID] = np.vstack((self.surfPoint[gridID],self.limitSurfacePP.gridCoord[gridID][coordinate]))
    if self.converged:
      self.raiseAMessage(self.name + " converged!")
    return ready

  def __scoreCandidates(self):
    """
      Compute the scores of the 'candidate set' which should be the currently
      extracted limit surface.
      @ In, None
      @ Out, None
    """
    # DM: This sequence gets used repetitively, so I am promoting it to its own
    #  variable
    axisNames = [key.replace('<distribution>','') for key in self.axisName]
    matrixShape = self.limitSurfacePP.getTestMatrix().shape
    self.scores = OrderedDict()
    if self.scoringMethod.startswith('distance'):
      sampledMatrix = np.zeros((len(self.limitSurfacePP.getFunctionValue()[axisNames[0]])+len(self.hangingPoints[:,0]),len(self.axisName)))
      for varIndex, name in enumerate(axisNames):
        sampledMatrix[:,varIndex] = np.append(self.limitSurfacePP.getFunctionValue()[name],self.hangingPoints[:,varIndex])
      distanceTree = spatial.cKDTree(copy.copy(sampledMatrix),leafsize=12)
      # The hanging point are added to the list of the already explored points
      # so as not to pick the same when in parallel
      for varIndex, _ in enumerate(axisNames):
        self.inputInfo['distributionName'][self.axisName[varIndex]] = self.toBeSampled[self.axisName[varIndex]]
        self.inputInfo['distributionType'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].type

      for key, value in self.invPointPersistence.items():
        if key != self.exceptionGrid and self.surfPoint[key] is not None:
          distance, _ = distanceTree.query(self.surfPoint[key])
          # Different versions of scipy/numpy will yield different results on
          # our various supported platforms. If things are this close, then it
          # it is highly unlikely choosing one point over the other will affect
          # us much, so limit the precision to allow the same results on older
          # versions. Scale could be important, though, so normalize the
          # distances first. Alternatively, we could force newer versions of
          # these libraries, but since our own HPC does not yet support them,
          # this should be acceptable, agreed? - DPM Nov. 23, 2015
          maxDistance = max(distance)
          if maxDistance != 0:
            distance = np.round(distance/maxDistance,15)
          if self.scoringMethod == 'distance' or max(self.invPointPersistence) == 0:
            self.scores[key] = distance
          else:
            self.scores[key] = np.multiply(distance,self.invPointPersistence[key])
    elif self.scoringMethod == 'debug':
      self.scores = OrderedDict()
      for key, value in self.invPointPersistence.items():
        self.scores[key] = np.zeros(len(self.surfPoint[key]))
        for i in range(len(self.listsurfPoint)):
          self.scores[key][i] = 1
    else:
      self.raiseAnError(NotImplementedError,self.scoringMethod + ' scoring method is not implemented yet')

  def localGenerateInput(self,model,oldInput):
    """
      Function to select the next most informative point for refining the limit
      surface search.
      After this method is called, the self.inputInfo should be ready to be sent
      to the model
      @ In, model, model instance, an instance of a model
      @ In, oldInput, list, a list of the original needed inputs for the model (e.g. list of files, etc.)
      @ Out, None
    """
    #  Alternatively, though I don't think we do this yet:
    #  compute the direction normal to the surface, compute the derivative
    #  normal to the surface of the probability, check the points where the
    #  derivative probability is the lowest

    # create values dictionary
    self.inputInfo['distributionName'] = {} #Used to determine which distribution to change if needed.
    self.inputInfo['distributionType'] = {} #Used to determine which distribution type is used
    self.raiseADebug('generating input')
    varSet=False

    # DM: This sequence gets used repetitively, so I am promoting it to its own
    #  variable
    axisNames = [key.replace('<distribution>','') for key in self.axisName]

    if self.surfPoint is not None and len(self.surfPoint) > 0:
      if self.batchStrategy == 'none':
        self.__scoreCandidates()
        maxDistance, maxGridId, maxId =  0.0, "", 0
        for key, value in sorted(self.invPointPersistence.items()):
          if key != self.exceptionGrid and self.surfPoint[key] is not None:
            localMax = np.max(self.scores[key])
            if localMax > maxDistance:
              maxDistance, maxGridId, maxId  = localMax, key,  np.argmax(self.scores[key])
        if maxDistance > 0.0:
          for varIndex, _ in enumerate([key.replace('<distribution>','') for key in self.axisName]):
            self.values[self.axisName[varIndex]] = copy.copy(float(self.surfPoint[maxGridId][maxId,varIndex]))
            self.inputInfo['SampledVarsPb'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
            self.inputInfo['ProbabilityWeight-'+self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
          varSet=True
        else:
          self.raiseADebug('Maximum score is 0.0')
      elif self.batchStrategy.startswith('max'):
        ########################################################################
        ## Initialize the queue with as many points as requested or as many as
        ## possible
        if len(self.toProcess) == 0:
          self.__scoreCandidates()
          edges = []

          flattenedSurfPoints = list()
          flattenedBandPoints = list()
          flattenedScores     = list()
          for key in self.bandIndices.keys():
            flattenedSurfPoints = flattenedSurfPoints + list(self.surfPoint[key])
            flattenedScores = flattenedScores + list(self.scores[key])
            flattenedBandPoints = flattenedBandPoints + self.listSurfPoint[key] + self.bandIndices[key]

          flattenedSurfPoints = np.array(flattenedSurfPoints)
          for i,iCoords in enumerate(flattenedBandPoints):
            for j in range(i+1, len(flattenedBandPoints)):
              jCoords = flattenedBandPoints[j]
              ijValidNeighbors = True
              for d in range(len(jCoords)):
                if abs(iCoords[d] - jCoords[d]) > 1:
                  ijValidNeighbors = False
                  break
              if ijValidNeighbors:
                edges.append((i,j))
                edges.append((j,i))

          names = axisNames[:] #make copy
          names.append('score')
          amsc = AMSC_Object(X=flattenedSurfPoints, Y=flattenedScores,
                             w=None, names=names, graph='none',
                             gradient='steepest', normalization='feature',
                             persistence='difference', edges=edges, debug=False)
          plevel = self.simplification*(max(flattenedScores)-min(flattenedScores))
          partitions = amsc.StableManifolds(plevel)
          mergeSequence = amsc.GetMergeSequence()
          maxIdxs = list(set(partitions.keys()))

          thresholdLevel = self.threshold*(max(flattenedScores)-min(flattenedScores))+min(flattenedScores)
          # Sort the maxima based on decreasing function value, thus the top
          # candidate is the first element.
          if self.batchStrategy.endswith('V'):
            sortedMaxima = sorted(maxIdxs, key=lambda idx: flattenedScores[idx], reverse=True)
          else:
          # Sort the maxima based on decreasing persistence value, thus the top
          # candidate is the first element.
            sortedMaxima = sorted(maxIdxs, key=lambda idx: mergeSequence[idx][1], reverse=True)
          B = min(self.maxBatchSize,len(sortedMaxima))
          for idx in sortedMaxima[0:B]:
            if flattenedScores[idx] >= thresholdLevel:
              self.toProcess.append(flattenedSurfPoints[idx,:])
          if len(self.toProcess) == 0:
            self.toProcess.append(flattenedSurfPoints[np.argmax(flattenedScores),:])
        ########################################################################
        ## Select one sample
        selectedPoint = self.toProcess.pop()
        for varIndex, varName in enumerate(axisNames):
          self.values[self.axisName[varIndex]] = float(selectedPoint[varIndex])
          self.inputInfo['SampledVarsPb'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
          self.inputInfo['ProbabilityWeight-'+self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
        varSet=True
      elif self.batchStrategy == 'naive':
        ########################################################################
        ## Initialize the queue with as many points as requested or as many as
        ## possible
        if len(self.toProcess) == 0:
          self.__scoreCandidates()
          sortedIndices = sorted(range(len(self.scores)), key=lambda k: self.scores[k],reverse=True)
          B = min(self.maxBatchSize,len(sortedIndices))
          for idx in sortedIndices[0:B]:
            self.toProcess.append(self.surfPoint[idx,:])
          if len(self.toProcess) == 0:
            self.toProcess.append(self.surfPoint[np.argmax(self.scores),:])
        ########################################################################
        ## Select one sample
        selectedPoint = self.toProcess.pop()
        for varIndex, varName in enumerate(axisNames):
          self.values[self.axisName[varIndex]] = float(selectedPoint[varIndex])
          self.inputInfo['SampledVarsPb'][self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
          self.inputInfo['ProbabilityWeight-'+self.axisName[varIndex]] = self.distDict[self.axisName[varIndex]].pdf(self.values[self.axisName[varIndex]])
        varSet=True

    if not varSet:
      #here we are still generating the batch
      for key in sorted(self.distDict.keys()):
        if self.toleranceWeight=='cdf':
          self.values[key]                       = self.distDict[key].ppf(float(randomUtils.random()))
        else:
          self.values[key]                       = self.distDict[key].lowerBound+(self.distDict[key].upperBound-self.distDict[key].lowerBound)*float(randomUtils.random())
        self.inputInfo['distributionName'][key]  = self.toBeSampled[key]
        self.inputInfo['distributionType'][key]  = self.distDict[key].type
        self.inputInfo['SampledVarsPb'   ][key]  = self.distDict[key].pdf(self.values[key])
        self.inputInfo['ProbabilityWeight-'+key] = self.distDict[key].pdf(self.values[key])
        self.addMetaKeys(['ProbabilityWeight-'+key])
    self.inputInfo['PointProbability'    ]      = reduce(mul, self.inputInfo['SampledVarsPb'].values())
    # the probability weight here is not used, the post processor is going to recreate the grid associated and use a ROM for the probability evaluation
    self.inputInfo['ProbabilityWeight']         = self.inputInfo['PointProbability']
    self.hangingPoints                          = np.vstack((self.hangingPoints,copy.copy(np.array([self.values[axis] for axis in self.axisName]))))
    self.raiseADebug('At counter '+str(self.counter)+' the generated sampled variables are: '+str(self.values))
    self.inputInfo['SamplerType'] = 'LimitSurfaceSearch'
    self.inputInfo['subGridTol' ] = self.subGridTol

    #      This is the normal derivation to be used later on
    #      pbMapPointCoord = np.zeros((len(self.surfPoint),self.nVar*2+1,self.nVar))
    #      for pointIndex, point in enumerate(self.surfPoint):
    #        temp = copy.copy(point)
    #        pbMapPointCoord[pointIndex,2*self.nVar,:] = temp
    #        for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
    #          temp[varIndex] -= np.max(self.axisStepSize[varName])
    #          pbMapPointCoord[pointIndex,varIndex,:] = temp
    #          temp[varIndex] += 2.*np.max(self.axisStepSize[varName])
    #          pbMapPointCoord[pointIndex,varIndex+self.nVar,:] = temp
    #          temp[varIndex] -= np.max(self.axisStepSize[varName])
    #      #getting the coordinate ready to be evaluated by the ROM
    #      pbMapPointCoord.shape = (len(self.surfPoint)*(self.nVar*2+1),self.nVar)
    #      tempDict = {}
    #      for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
    #        tempDict[varName] = pbMapPointCoord.T[varIndex,:]
    #      #acquiring Pb evaluation
    #      pbPoint       = self.ROM.confidence(tempDict)
    #      pbPoint.shape = (len(self.surfPoint),self.nVar*2+1,2)
    #      pbMapPointCoord.shape = (len(self.surfPoint),self.nVar*2+1,self.nVar)
    #      #computing gradient
    #      modGrad   = np.zeros((len(self.surfPoint)))
    #      gradVect  = np.zeros((len(self.surfPoint),self.nVar))
    #      for pointIndex in range(len(self.surfPoint)):
    #        centralCoor = pbMapPointCoord[pointIndex,2*self.nVar,:]
    #        centraPb    = pbPoint[pointIndex,2*self.nVar][0]
    #        sum = 0.0
    #        for varIndex in range(self.nVar):
    #          d1Down     = (centraPb-pbPoint[pointIndex,varIndex][0])/(centralCoor[varIndex]-pbMapPointCoord[pointIndex,varIndex,varIndex])
    #          d1Up       = (pbPoint[pointIndex,varIndex+self.nVar][0]-centraPb)/(pbMapPointCoord[pointIndex,varIndex+self.nVar,varIndex]-centralCoor[varIndex])
    #          if np.abs(d1Up)>np.abs(d1Down): d1Avg = d1Up
    #          else                          : d1Avg = d1Down
    #          gradVect[pointIndex,varIndex] = d1Avg
    #          sum +=d1Avg
    #          modGrad[pointIndex] += d1Avg**2
    #        modGrad[pointIndex] = np.sqrt(modGrad[pointIndex])*np.abs(sum)/sum
    #        #concavityPb[pointIndex] = concavityPb[pointIndex]/float(self.nVar)
    #      for pointIndex, point in enumerate(self.surfPoint):
    #        myStr  = ''
    #        myStr  += '['
    #        for varIndex in range(self.nVar):
    #          myStr += '{:+6.4f}'.format(pbMapPointCoord[pointIndex,2*self.nVar,varIndex])
    #        myStr += '] '+'{:+6.4f}'.format(pbPoint[pointIndex,2*self.nVar,0])+'   '
    #        for varIndex in range(2*self.nVar):
    #          myStr += '['
    #          for varIndex2 in range(self.nVar):
    #            myStr += '{:+6.4f}'.format(pbMapPointCoord[pointIndex,varIndex,varIndex2])+' '
    #          myStr += '] '+'{:+6.4f}'.format(pbPoint[pointIndex,varIndex,0])+'   '
    #        myStr += '   gradient  ['
    #        for varIndex in range(self.nVar):
    #          myStr += '{:+6.4f}'.format(gradVect[pointIndex,varIndex])+'  '
    #        myStr += ']'
    #        myStr += '    Module '+'{:+6.4f}'.format(modGrad[pointIndex])
    #
    #      minIndex = np.argmin(np.abs(modGrad))
    #      pdDist = self.sign*(pbPoint[minIndex,2*self.nVar][0]-0.5-10*self.tolerance)/modGrad[minIndex]
    #      for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
    #        self.values[varName] = copy.copy(float(pbMapPointCoord[minIndex,2*self.nVar,varIndex]+pdDist*gradVect[minIndex,varIndex]))
    #      gradVect = np.ndarray(self.nVar)
    #      centraPb = pbPoint[minIndex,2*self.nVar]
    #      centralCoor = pbMapPointCoord[minIndex,2*self.nVar,:]
    #      for varIndex in range(self.nVar):
    #        d1Down = (centraPb-pbPoint[minIndex,varIndex])/(centralCoor[varIndex]-pbMapPointCoord[minIndex,varIndex,varIndex])
    #        d1Up   = (pbPoint[minIndex,varIndex+self.nVar]-centraPb)/(pbMapPointCoord[minIndex,varIndex+self.nVar,varIndex]-centralCoor[varIndex])
    #        d1Avg   = (d1Up+d1Down)/2.0
    #        gradVect[varIndex] = d1Avg
    #      gradVect = gradVect*pdDist
    #      gradVect = gradVect+centralCoor
    #      for varIndex, varName in enumerate([key.replace('<distribution>','') for key in self.axisName]):
    #        self.values[varName] = copy.copy(float(gradVect[varIndex]))

  def _formatSolutionExportVariableNames(self, acceptable):
    """
      Does magic formatting for variables, based on this class's needs.
      Extend in inheritors as needed.
      @ In, acceptable, set, set of acceptable entries for solution export for this entity
      @ Out, new, set, modified set of acceptable variables with all formatting complete
    """
    # remaking the list is easier than using the existing one
    acceptable = AdaptiveSampler._formatSolutionExportVariableNames(self, acceptable)
    new = []
    while acceptable:
      template = acceptable.pop()
      if template == '{RESIDUUM}':
        new.append(template.format(RESIDUUM=self.goalFunction.name))
      else:
        new.append(template)
    return set(new)
