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
from __future__ import division, print_function , unicode_literals, absolute_import
import warnings
warnings.simplefilter('default', DeprecationWarning)

#External Modules------------------------------------------------------------------------------------
import numpy as np
from scipy import spatial, interpolate
import os
from glob import glob
import copy
import math
from collections import OrderedDict, defaultdict
import time
from sklearn.linear_model import LinearRegression
import importlib
import abc
import six
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from BaseClasses import BaseType
from utils import utils
from utils import mathUtils
from utils import xmlUtils
from utils import InputData
from utils.RAVENiterators import ravenArrayIterator
import DataObjects
from Assembler import Assembler
import LearningGate
import MessageHandler
import GridEntities
import Files
import Models
import unSupervisedLearning
from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase
#Internal Modules End--------------------------------------------------------------------------------

#
#  ***************************************
#  *  SPECIALIZED PostProcessor CLASSES  *
#  ***************************************
#

##########################################################
## Temporary addition, remove this code once this inherits
## from the base type
class ModelInput(InputData.ParameterInput):
  """
    Class for reading in model input
  """

ModelInput.createClass("ModelInput")
ModelInput.addParam("subType", InputData.StringType, True)
##########################################################

class BasePostProcessor(Assembler, MessageHandler.MessageUser):
  """
    This is the base class for postprocessors
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    Assembler.__init__(self)
    self.type = self.__class__.__name__  # pp type
    self.name = self.__class__.__name__  # pp name
    self.messageHandler = messageHandler

  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    # if 'externalFunction' in initDict.keys(): self.externalFunction = initDict['externalFunction']
    self.inputs = inputs

  def inputToInternal(self, currentInput):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputToInternal, list, list of current inputs
    """
    return [(copy.deepcopy(currentInput))]

  def run(self, input):
    """
      This method executes the postprocessor action.
      @ In,  input, object, object containing the data to process. (inputToInternal output)
      @ Out, None
    """
    pass

class LimitSurfaceIntegralInput(InputData.ParameterInput):
  """
    Class for reading in the limit surface integral block
  """

LimitSurfaceIntegralInput.createClass("PostProcessor", False, baseNode=ModelInput)
LSIVariableInput = InputData.parameterInputFactory("variable")
LSIVariableInput.addParam("name", InputData.StringType)
LSIDistributionInput = InputData.parameterInputFactory("distribution", contentType=InputData.StringType)
LSIVariableInput.addSub(LSIDistributionInput)
LSILowerBoundInput = InputData.parameterInputFactory("lowerBound", contentType=InputData.FloatType)
LSIVariableInput.addSub(LSILowerBoundInput)
LSIUpperBoundInput = InputData.parameterInputFactory("upperBound", contentType=InputData.FloatType)
LSIVariableInput.addSub(LSIUpperBoundInput)
LimitSurfaceIntegralInput.addSub(LSIVariableInput)
LSIToleranceInput = InputData.parameterInputFactory("tolerance", contentType=InputData.FloatType)
LimitSurfaceIntegralInput.addSub(LSIToleranceInput)
LSIIntegralTypeInput = InputData.parameterInputFactory("integralType", contentType=InputData.StringType)
LimitSurfaceIntegralInput.addSub(LSIIntegralTypeInput)
LSISeedInput = InputData.parameterInputFactory("seed", contentType=InputData.IntegerType)
LimitSurfaceIntegralInput.addSub(LSISeedInput)
LSITargetInput = InputData.parameterInputFactory("target", contentType=InputData.StringType)
LimitSurfaceIntegralInput.addSub(LSITargetInput)

class LimitSurfaceIntegral(BasePostProcessor):
  """
    This post-processor computes the n-dimensional integral of a Limit Surface
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self, messageHandler)
    self.variableDist = {}  # dictionary created upon the .xml input file reading. It stores the distributions for each variable.
    self.target = None  # target that defines the f(x1,x2,...,xn)
    self.tolerance = 0.0001  # integration tolerance
    self.integralType = 'montecarlo'  # integral type (which alg needs to be used). Either montecarlo or quadrature(quadrature not yet)
    self.seed = 20021986  # seed for montecarlo
    self.matrixDict = {}  # dictionary of arrays and target
    self.lowerUpperDict = {}
    self.functionS = None
    self.stat = returnInstance('BasicStatistics', self)  # instantiation of the 'BasicStatistics' processor, which is used to compute the pb given montecarlo evaluations
    self.stat.what = ['expectedValue']
    self.addAssemblerObject('Distribution','n', newXmlFlg = False)
    self.printTag = 'POSTPROCESSOR INTEGRAL'

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by this postprocessor that need to request special objects
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    needDict = {'Distributions':[]}
    for distName in self.variableDist.values():
      if distName != None: needDict['Distributions'].append((None, distName))
    return needDict

  def _localGenerateAssembler(self, initDict):
    """
      This method  is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    for varName, distName in self.variableDist.items():
      if distName != None:
        if distName not in initDict['Distributions'].keys(): self.raiseAnError(IOError, 'distribution ' + distName + ' not found.')
        self.variableDist[varName] = initDict['Distributions'][distName]
        self.lowerUpperDict[varName]['lowerBound'] = self.variableDist[varName].lowerBound
        self.lowerUpperDict[varName]['upperBound'] = self.variableDist[varName].upperBound

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = LimitSurfaceIntegralInput()
    paramInput.parseNode(xmlNode)
    for child in paramInput.subparts:
      varName = None
      if child.getName() == 'variable':
        varName = child.parameterValues['name']
        self.lowerUpperDict[varName] = {}
        self.variableDist[varName] = None
        for childChild in child.subparts:
          if childChild.getName() == 'distribution':
            self.variableDist[varName] = childChild.value
          elif childChild.getName() == 'lowerBound':
            if self.variableDist[varName] != None:
              self.raiseAnError(NameError, 'you can not specify both distribution and lower/upper bounds nodes for variable ' + varName + ' !')
            self.lowerUpperDict[varName]['lowerBound'] = childChild.value
          elif childChild.getName() == 'upperBound':
            if self.variableDist[varName] != None:
              self.raiseAnError(NameError, 'you can not specify both distribution and lower/upper bounds nodes for variable ' + varName + ' !')
            self.lowerUpperDict[varName]['upperBound'] = childChild.value
          else:
            self.raiseAnError(NameError, 'invalid labels after the variable call. Only "distribution", "lowerBound" abd "upperBound" is accepted. tag: ' + child.getName())
      elif child.getName() == 'tolerance':
        try:
          self.tolerance = child.value
        except ValueError:
          self.raiseAnError(ValueError, "tolerance can not be converted into a float value!")
      elif child.getName() == 'integralType':
        self.integralType = child.value.strip().lower()
        if self.integralType not in ['montecarlo']:
          self.raiseAnError(IOError, 'only one integral types are available: MonteCarlo!')
      elif child.getName() == 'seed':
        try:
          self.seed = child.value
        except ValueError:
          self.raiseAnError(ValueError, 'seed can not be converted into a int value!')
        if self.integralType != 'montecarlo':
          self.raiseAWarning('integral type is ' + self.integralType + ' but a seed has been inputted!!!')
        else:
          np.random.seed(self.seed)
      elif child.getName() == 'target':
        self.target = child.value
      else:
        self.raiseAnError(NameError, 'invalid or missing labels after the variables call. Only "variable" is accepted.tag: ' + child.getName())
      # if no distribution, we look for the integration domain in the input
      if varName != None:
        if self.variableDist[varName] == None:
          if 'lowerBound' not in self.lowerUpperDict[varName].keys() or 'upperBound' not in self.lowerUpperDict[varName].keys():
            self.raiseAnError(NameError, 'either a distribution name or lowerBound and upperBound need to be specified for variable ' + varName)
    if self.target == None: self.raiseAWarning('integral target has not been provided. The postprocessor is going to take the last output it finds in the provided limitsurface!!!')

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    self.inputToInternal(inputs)
    if self.integralType in ['montecarlo']:
      self.stat.toDo = {'expectedValue':set([self.target])}
      self.stat.initialize(runInfo, inputs, initDict)
    self.functionS = LearningGate.returnInstance('SupervisedGate','SciKitLearn', self, **{'SKLtype':'neighbors|KNeighborsClassifier', 'Features':','.join(list(self.variableDist.keys())), 'Target':self.target})
    self.functionS.train(self.matrixDict)
    self.raiseADebug('DATA SET MATRIX:')
    self.raiseADebug(self.matrixDict)

  def inputToInternal(self, currentInput):
    """
     Method to convert an input object into the internal format that is
     understandable by this pp.
     The resulting converted object is stored as an attribute of this class
     @ In, currentInput, object, an object that needs to be converted
     @ Out, None
    """
    for item in currentInput:
      if item.type == 'PointSet':
        self.matrixDict = {}
        if not set(item.getParaKeys('inputs')) == set(self.variableDist.keys()): self.raiseAnError(IOError, 'The variables inputted and the features in the input PointSet ' + item.name + 'do not match!!!')
        if self.target == None: self.target = item.getParaKeys('outputs')[-1]
        if self.target not in item.getParaKeys('outputs'): self.raiseAnError(IOError, 'The target ' + self.target + 'is not present among the outputs of the PointSet ' + item.name)
        # construct matrix
        for  varName in self.variableDist.keys(): self.matrixDict[varName] = item.getParam('input', varName)
        outputarr = item.getParam('output', self.target)
        if len(set(outputarr)) != 2: self.raiseAnError(IOError, 'The target ' + self.target + ' needs to be a classifier output (-1 +1 or 0 +1)!')
        outputarr[outputarr == -1] = 0.0
        self.matrixDict[self.target] = outputarr
      else: self.raiseAnError(IOError, 'Only PointSet is accepted as input!!!!')

  def run(self, input):
    """
      This method executes the postprocessor action. In this case, it performs the computation of the LS integral
      @ In,  input, object, object contained the data to process. (inputToInternal output)
      @ Out, pb, float, integral outcome (probability of the event)
    """
    pb = None
    if self.integralType == 'montecarlo':
      tempDict = {}
      randomMatrix = np.random.rand(int(math.ceil(1.0 / self.tolerance**2)), len(self.variableDist.keys()))
      for index, varName in enumerate(self.variableDist.keys()):
        if self.variableDist[varName] == None: randomMatrix[:, index] = randomMatrix[:, index] * (self.lowerUpperDict[varName]['upperBound'] - self.lowerUpperDict[varName]['lowerBound']) + self.lowerUpperDict[varName]['lowerBound']
        else:
          f = np.vectorize(self.variableDist[varName].ppf, otypes=[np.float])
          randomMatrix[:, index] = f(randomMatrix[:, index])
        tempDict[varName] = randomMatrix[:, index]
      pb = self.stat.run({'targets':{self.target:self.functionS.evaluate(tempDict)[self.target]}})['expectedValue'][self.target]
    else: self.raiseAnError(NotImplemented, "quadrature not yet implemented")
    return pb

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1: self.raiseAnError(RuntimeError, 'no available output to collect.')
    else:
      pb = finishedJob.getEvaluation()[1]
      lms = finishedJob.getEvaluation()[0][0]
      if output.type == 'PointSet':
        # we store back the limitsurface
        for key, value in lms.getParametersValues('input').items():
          for val in value: output.updateInputValue(key, val)
        for key, value in lms.getParametersValues('output').items():
          for val in value: output.updateOutputValue(key, val)
        for _ in range(len(lms)): output.updateOutputValue('EventProbability', pb)
      elif isinstance(output,Files.File):
        headers = lms.getParaKeys('inputs') + lms.getParaKeys('outputs')
        if 'EventProbability' not in headers: headers += ['EventProbability']
        stack = [None] * len(headers)
        output.close()
        outIndex = 0
        for key, value in lms.getParametersValues('input').items() : stack[headers.index(key)] = np.asarray(value).flatten()
        for key, value in lms.getParametersValues('output').items():
          stack[headers.index(key)] = np.asarray(value).flatten()
          outIndex = headers.index(key)
        stack[headers.index('EventProbability')] = np.array([pb] * len(stack[outIndex])).flatten()
        stacked = np.column_stack(stack)
        np.savetxt(output, stacked, delimiter = ',', header = ','.join(headers),comments='')
        #N.B. without comments='' you get a "# " at the top of the header row
      else: self.raiseAnError(Exception, self.type + ' accepts PointSet or File type only')
#
#

class SafestPointInput(InputData.ParameterInput):
  """
    class for reading in the Safest Point block
  """

SafestPointInput.createClass("PostProcessor", False, baseNode=ModelInput)
OuterDistributionInput = InputData.parameterInputFactory("Distribution", contentType=InputData.StringType)
OuterDistributionInput.addParam("class", InputData.StringType)
OuterDistributionInput.addParam("type", InputData.StringType)
SafestPointInput.addSub(OuterDistributionInput)
VariableInput = InputData.parameterInputFactory("variable")
VariableInput.addParam("name", InputData.StringType)
InnerDistributionInput = InputData.parameterInputFactory("distribution", contentType=InputData.StringType)
VariableInput.addSub(InnerDistributionInput)
InnerGridInput = InputData.parameterInputFactory("grid", contentType=InputData.FloatType)
InnerGridInput.addParam("type", InputData.StringType)
InnerGridInput.addParam("steps", InputData.IntegerType)
VariableInput.addSub(InnerGridInput)
ControllableInput = InputData.parameterInputFactory("controllable", contentType=InputData.StringType)
ControllableInput.addSub(VariableInput)
SafestPointInput.addSub(ControllableInput)
NoncontrollableInput = InputData.parameterInputFactory("non-controllable", contentType=InputData.StringType)
NoncontrollableInput.addSub(VariableInput)
SafestPointInput.addSub(NoncontrollableInput)

#
class SafestPoint(BasePostProcessor):
  """
    It searches for the probability-weighted safest point inside the space of the system controllable variables
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self, messageHandler)
    self.controllableDist = {}  # dictionary created upon the .xml input file reading. It stores the distributions for each controllable variable.
    self.nonControllableDist = {}  # dictionary created upon the .xml input file reading. It stores the distributions for each non-controllable variable.
    self.controllableGrid = {}  # dictionary created upon the .xml input file reading. It stores the grid type ('value' or 'CDF'), the number of steps and the step length for each controllale variable.
    self.nonControllableGrid = {}  # dictionary created upon the .xml input file reading. It stores the grid type ('value' or 'CDF'), the number of steps and the step length for each non-controllale variable.
    self.gridInfo = {}  # dictionary containing the grid type ('value' or 'CDF'), the grid construction type ('equal', set by default) and the list of sampled points for each variable.
    self.controllableOrd = []  # list containing the controllable variables' names in the same order as they appear inside the controllable space (self.controllableSpace)
    self.nonControllableOrd = []  # list containing the controllable variables' names in the same order as they appear inside the non-controllable space (self.nonControllableSpace)
    self.surfPointsMatrix = None  # 2D-matrix containing the coordinates of the points belonging to the failure boundary (coordinates are derived from both the controllable and non-controllable space)
    self.stat = returnInstance('BasicStatistics', self)  # instantiation of the 'BasicStatistics' processor, which is used to compute the expected value of the safest point through the coordinates and probability values collected in the 'run' function
    self.addAssemblerObject('Distribution','n', True)
    self.printTag = 'POSTPROCESSOR SAFESTPOINT'

  def _localGenerateAssembler(self, initDict):
    """
      This method  is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    for varName, distName in self.controllableDist.items():
      if distName not in initDict['Distributions'].keys():
        self.raiseAnError(IOError, 'distribution ' + distName + ' not found.')
      self.controllableDist[varName] = initDict['Distributions'][distName]
    for varName, distName in self.nonControllableDist.items():
      if distName not in initDict['Distributions'].keys():
        self.raiseAnError(IOError, 'distribution ' + distName + ' not found.')
      self.nonControllableDist[varName] = initDict['Distributions'][distName]

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = SafestPointInput()
    paramInput.parseNode(xmlNode)
    for child in paramInput.subparts:
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
    self.__gridSetting__()
    self.__gridGeneration__()
    self.inputToInternal(inputs)
    #FIXME this is quite invasive use of the basic statistics; a more standardized construction would be nice
    #we set the toDo here, since at this point we know the targets for the basic statistics
    self.stat.toDo = {'expectedValue':set(self.controllableOrd)} #don't set directly, just set up the toDo for basicStats
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
    for item in currentInput:
      if item.type == 'PointSet':
        self.surfPointsMatrix = np.zeros((len(item.getParam('output', item.getParaKeys('outputs')[-1])), len(self.gridInfo.keys()) + 1))
        k = 0
        for varName in self.controllableOrd:
          self.surfPointsMatrix[:, k] = item.getParam('input', varName)
          k += 1
        for varName in self.nonControllableOrd:
          self.surfPointsMatrix[:, k] = item.getParam('input', varName)
          k += 1
        self.surfPointsMatrix[:, k] = item.getParam('output', item.getParaKeys('outputs')[-1])

  def run(self, input):
    """
      This method executes the postprocessor action. In this case, it computes the safest point
      @ In,  input, object, object contained the data to process. (inputToInternal output)
      @ Out, dataCollector, PointSet, PointSet containing the elaborated data
    """
    nearestPointsInd = []
    dataCollector = DataObjects.returnInstance('PointSet', self)
    dataCollector.type = 'PointSet'
    surfTree = spatial.KDTree(copy.copy(self.surfPointsMatrix[:, 0:self.surfPointsMatrix.shape[-1] - 1]))
    self.controllableSpace.shape = (np.prod(self.controllableSpace.shape[0:len(self.controllableSpace.shape) - 1]), self.controllableSpace.shape[-1])
    self.nonControllableSpace.shape = (np.prod(self.nonControllableSpace.shape[0:len(self.nonControllableSpace.shape) - 1]), self.nonControllableSpace.shape[-1])
    self.raiseADebug('RESHAPED CONTROLLABLE SPACE:')
    self.raiseADebug(self.controllableSpace)
    self.raiseADebug('RESHAPED NON-CONTROLLABLE SPACE:')
    self.raiseADebug(self.nonControllableSpace)
    for ncLine in range(self.nonControllableSpace.shape[0]):
      queryPointsMatrix = np.append(self.controllableSpace, np.tile(self.nonControllableSpace[ncLine, :], (self.controllableSpace.shape[0], 1)), axis = 1)
      self.raiseADebug('QUERIED POINTS MATRIX:')
      self.raiseADebug(queryPointsMatrix)
      nearestPointsInd = surfTree.query(queryPointsMatrix)[-1]
      distList = []
      indexList = []
      probList = []
      for index in range(len(nearestPointsInd)):
        if self.surfPointsMatrix[np.where(np.prod(surfTree.data[nearestPointsInd[index], 0:self.surfPointsMatrix.shape[-1] - 1] == self.surfPointsMatrix[:, 0:self.surfPointsMatrix.shape[-1] - 1], axis = 1))[0][0], -1] == 1:
          distList.append(np.sqrt(np.sum(np.power(queryPointsMatrix[index, 0:self.controllableSpace.shape[-1]] - surfTree.data[nearestPointsInd[index], 0:self.controllableSpace.shape[-1]], 2))))
          indexList.append(index)
      if distList == []:
        self.raiseAnError(ValueError, 'no safest point found for the current set of non-controllable variables: ' + str(self.nonControllableSpace[ncLine, :]) + '.')
      else:
        for cVarIndex in range(len(self.controllableOrd)):
          dataCollector.updateInputValue(self.controllableOrd[cVarIndex], copy.copy(queryPointsMatrix[indexList[distList.index(max(distList))], cVarIndex]))
        for ncVarIndex in range(len(self.nonControllableOrd)):
          dataCollector.updateInputValue(self.nonControllableOrd[ncVarIndex], copy.copy(queryPointsMatrix[indexList[distList.index(max(distList))], len(self.controllableOrd) + ncVarIndex]))
          if queryPointsMatrix[indexList[distList.index(max(distList))], len(self.controllableOrd) + ncVarIndex] == self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].lowerBound:
            if self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][0] == 'CDF':
              prob = self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / float(2)
            else:
              prob = self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].lowerBound + self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / float(2))
          elif queryPointsMatrix[indexList[distList.index(max(distList))], len(self.controllableOrd) + ncVarIndex] == self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].upperBound:
            if self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][0] == 'CDF':
              prob = self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / float(2)
            else:
              prob = 1 - self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].upperBound - self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / float(2))
          else:
            if self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][0] == 'CDF':
              prob = self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2]
            else:
              prob = self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(queryPointsMatrix[indexList[distList.index(max(distList))], len(self.controllableOrd) + ncVarIndex] + self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / float(2)) - self.nonControllableDist[self.nonControllableOrd[ncVarIndex]].cdf(queryPointsMatrix[indexList[distList.index(max(distList))], len(self.controllableOrd) + ncVarIndex] - self.nonControllableGrid[self.nonControllableOrd[ncVarIndex]][2] / float(2))
          probList.append(prob)
      dataCollector.updateOutputValue('Probability', np.prod(probList))
      dataCollector.updateMetadata('ProbabilityWeight', np.prod(probList))
    dataCollector.updateMetadata('ExpectedSafestPointCoordinates', self.stat.run(dataCollector)['expectedValue'])
    self.raiseADebug(dataCollector.getParametersValues('input'))
    self.raiseADebug(dataCollector.getParametersValues('output'))
    self.raiseADebug(dataCollector.getMetadata('ExpectedSafestPointCoordinates'))
    return dataCollector

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1:
      self.raiseAnError(RuntimeError, 'no available output to collect (the run is likely not over yet).')
    else:
      dataCollector = finishedJob.getEvaluation()[1]
      if output.type != 'PointSet':
        self.raiseAnError(TypeError, 'output item type must be "PointSet".')
      else:
        if not output.isItEmpty():
          self.raiseAnError(ValueError, 'output item must be empty.')
        else:
          for key, value in dataCollector.getParametersValues('input').items():
            for val in value: output.updateInputValue(key, val)
          for key, value in dataCollector.getParametersValues('output').items():
            for val in value: output.updateOutputValue(key, val)
          for key, value in dataCollector.getAllMetadata().items(): output.updateMetadata(key, value)
#
#

class ComparisonStatisticsInput(InputData.ParameterInput):
  """
    class for reading in comparison statistics block
  """

ComparisonStatisticsInput.createClass("PostProcessor", False, baseNode=ModelInput)
KindInputEnumType = InputData.makeEnumType("kind","kindType",["uniformBins","equalProbability"])
KindInput = InputData.parameterInputFactory("kind", contentType=KindInputEnumType)
KindInput.addParam("numBins",InputData.IntegerType, False)
KindInput.addParam("binMethod", InputData.StringType, False)
ComparisonStatisticsInput.addSub(KindInput)
class CSCompareInput(InputData.ParameterInput):
  """
    class for reading in the compare block in comparison statistics
  """

CSCompareInput.createClass("compare", False)
CSDataInput = InputData.parameterInputFactory("data", contentType=InputData.StringType)
CSCompareInput.addSub(CSDataInput)
CSReferenceInput = InputData.parameterInputFactory("reference")
CSReferenceInput.addParam("name", InputData.StringType, True)
CSCompareInput.addSub(CSReferenceInput)
ComparisonStatisticsInput.addSub(CSCompareInput)
FZInput = InputData.parameterInputFactory("fz", contentType=InputData.StringType) #bool
ComparisonStatisticsInput.addSub(FZInput)
CSInterpolationEnumType = InputData.makeEnumType("csinterpolation","csinterpolationType",["linear","quadratic"])
CSInterpolationInput = InputData.parameterInputFactory("interpolation",contentType=CSInterpolationEnumType)
ComparisonStatisticsInput.addSub(CSInterpolationInput)

#
class ComparisonStatistics(BasePostProcessor):
  """
    ComparisonStatistics is to calculate statistics that compare
    two different codes or code to experimental data.
  """

  class CompareGroup:
    """
      Class aimed to compare two group of data
    """
    def __init__(self):
      """
        Constructor
        @ In, None
        @ Out, None
      """
      self.dataPulls = []
      self.referenceData = {}

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self, messageHandler)
    self.dataDict = {}  # Dictionary of all the input data, keyed by the name
    self.compareGroups = []  # List of each of the groups that will be compared
    # self.dataPulls = [] #List of data references that will be used
    # self.referenceData = [] #List of reference (experimental) data
    self.methodInfo = {}  # Information on what stuff to do.
    self.fZStats = False
    self.interpolation = "quadratic"
    self.requiredAssObject = (True, (['Distribution'], ['-n']))
    self.distributions = {}

  def inputToInternal(self, currentInput):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputToInternal, list, the resulting converted object
    """
    return [(currentInput)]

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the ComparisonStatistics pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = ComparisonStatisticsInput()
    paramInput.parseNode(xmlNode)
    for outer in paramInput.subparts:
      if outer.getName() == 'compare':
        compareGroup = ComparisonStatistics.CompareGroup()
        for child in outer.subparts:
          if child.getName() == 'data':
            dataName = child.value
            splitName = dataName.split("|")
            name, kind = splitName[:2]
            rest = splitName[2:]
            compareGroup.dataPulls.append([name, kind, rest])
          elif child.getName() == 'reference':
            # This has name=distribution
            compareGroup.referenceData = dict(child.parameterValues)
            if "name" not in compareGroup.referenceData:
              self.raiseAnError(IOError, 'Did not find name in reference block')

        self.compareGroups.append(compareGroup)
      if outer.getName() == 'kind':
        self.methodInfo['kind'] = outer.value
        if 'numBins' in outer.parameterValues:
          self.methodInfo['numBins'] = outer.parameterValues['numBins']
        if 'binMethod' in outer.parameterValues:
          self.methodInfo['binMethod'] = outer.parameterValues['binMethod'].lower()
      if outer.getName() == 'fz':
        self.fZStats = (outer.value.lower() in utils.stringsThatMeanTrue())
      if outer.getName() == 'interpolation':
        interpolation = outer.value.lower()
        if interpolation == 'linear':
          self.interpolation = 'linear'
        elif interpolation == 'quadratic':
          self.interpolation = 'quadratic'
        else:
          self.raiseADebug('unexpected interpolation method ' + interpolation)
          self.interpolation = interpolation


  def _localGenerateAssembler(self, initDict):
    """
      This method  is used for sending to the instanciated class, which is implementing the method, the objects that have been requested through "whatDoINeed" method
      It is an abstract method -> It must be implemented in the derived class!
      @ In, initDict, dict, dictionary ({'mainClassName(e.g., Databases):{specializedObjectName(e.g.,DatabaseForSystemCodeNamedWolf):ObjectInstance}'})
      @ Out, None
    """
    self.distributions = initDict.get('Distributions', {})

  def run(self, input):  # inObj,workingDir=None):
    """
      This method executes the postprocessor action. In this case, it just returns the inputs
      @ In,  input, object, object contained the data to process. (inputToInternal output)
      @ Out, dataDict, dict, Dictionary containing the inputs
    """
    dataDict = {}
    for aInput in input:
      dataDict[aInput.name] = aInput
    return dataDict

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    self.raiseADebug("finishedJob: " + str(finishedJob) + ", output " + str(output))
    if finishedJob.getEvaluation() == -1: self.raiseAnError(RuntimeError, 'no available output to collect.')
    else: self.dataDict.update(finishedJob.getEvaluation()[1])

    dataToProcess = []
    for compareGroup in self.compareGroups:
      dataPulls = compareGroup.dataPulls
      reference = compareGroup.referenceData
      foundDataObjects = []
      for name, kind, rest in dataPulls:
        data = self.dataDict[name].getParametersValues(kind)
        if len(rest) == 1:
          foundDataObjects.append(data[rest[0]])
      dataToProcess.append((dataPulls, foundDataObjects, reference))
    generateCSV = False
    generatePointSet = False
    if isinstance(output,Files.File):
      generateCSV = True
    elif output.type == 'PointSet':
      generatePointSet = True
    else:
      self.raiseAnError(IOError, 'unsupported type ' + str(type(output)))
    if generateCSV:
      csv = output
    for dataPulls, datas, reference in dataToProcess:
      graphData = []
      if "name" in reference:
        distributionName = reference["name"]
        if not distributionName in self.distributions:
          self.raiseAnError(IOError, 'Did not find ' + distributionName +
                             ' in ' + str(self.distributions.keys()))
        else:
          distribution = self.distributions[distributionName]
        refDataStats = {"mean":distribution.untruncatedMean(),
                        "stdev":distribution.untruncatedStdDev()}
        refDataStats["minBinSize"] = refDataStats["stdev"] / 2.0
        refPdf = lambda x:distribution.pdf(x)
        refCdf = lambda x:distribution.cdf(x)
        graphData.append((refDataStats, refCdf, refPdf, "ref_" + distributionName))
      for dataPull, data in zip(dataPulls, datas):
        dataStats = self.__processData( data, self.methodInfo)
        dataKeys = set(dataStats.keys())
        counts = dataStats['counts']
        bins = dataStats['bins']
        countSum = sum(counts)
        binBoundaries = [dataStats['low']] + bins + [dataStats['high']]
        if generateCSV:
          utils.printCsv(csv, '"' + str(dataPull) + '"')
          utils.printCsv(csv, '"numBins"', dataStats['numBins'])
          utils.printCsv(csv, '"binBoundary"', '"binMidpoint"', '"binCount"', '"normalizedBinCount"', '"f_prime"', '"cdf"')
        cdf = [0.0] * len(counts)
        midpoints = [0.0] * len(counts)
        cdfSum = 0.0
        for i in range(len(counts)):
          f0 = counts[i] / countSum
          cdfSum += f0
          cdf[i] = cdfSum
          midpoints[i] = (binBoundaries[i] + binBoundaries[i + 1]) / 2.0
        cdfFunc = mathUtils.createInterp(midpoints, cdf, 0.0, 1.0, self.interpolation)
        fPrimeData = [0.0] * len(counts)
        for i in range(len(counts)):
          h = binBoundaries[i + 1] - binBoundaries[i]
          nCount = counts[i] / countSum  # normalized count
          f0 = cdf[i]
          if i + 1 < len(counts):
            f1 = cdf[i + 1]
          else:
            f1 = 1.0
          if i + 2 < len(counts):
            f2 = cdf[i + 2]
          else:
            f2 = 1.0
          if self.interpolation == 'linear':
            fPrime = (f1 - f0) / h
          else:
            fPrime = (-1.5 * f0 + 2.0 * f1 + -0.5 * f2) / h
          fPrimeData[i] = fPrime
          if generateCSV:
            utils.printCsv(csv, binBoundaries[i + 1], midpoints[i], counts[i], nCount, fPrime, cdf[i])
        pdfFunc = mathUtils.createInterp(midpoints, fPrimeData, 0.0, 0.0, self.interpolation)
        dataKeys -= set({'numBins', 'counts', 'bins'})
        if generateCSV:
          for key in dataKeys:
            utils.printCsv(csv, '"' + key + '"', dataStats[key])
        self.raiseADebug("dataStats: " + str(dataStats))
        graphData.append((dataStats, cdfFunc, pdfFunc, str(dataPull)))
      graphDataDict = mathUtils.getGraphs(graphData, self.fZStats)
      if generateCSV:
        for key in graphDataDict:
          value = graphDataDict[key]
          if type(value).__name__ == 'list':
            utils.printCsv(csv, *(['"' + l[0] + '"' for l in value]))
            for i in range(1, len(value[0])):
              utils.printCsv(csv, *([l[i] for l in value]))
          else:
            utils.printCsv(csv, '"' + key + '"', value)
      if generatePointSet:
        for key in graphDataDict:
          value = graphDataDict[key]
          if type(value).__name__ == 'list':
            for i in range(len(value)):
              subvalue = value[i]
              name = subvalue[0]
              subdata = subvalue[1:]
              if i == 0:
                output.updateInputValue(name, subdata)
              else:
                output.updateOutputValue(name, subdata)
            break  # XXX Need to figure out way to specify which data to return
      if generateCSV:
        for i in range(len(graphData)):
          dataStat = graphData[i][0]
          def delist(l):
            """
              Method to create a string out of a list l
              @ In, l, list, the list to be 'stringed' out
              @ Out, delist, string, the string representing the list
            """
            if type(l).__name__ == 'list':
              return '_'.join([delist(x) for x in l])
            else:
              return str(l)
          newFileName = output.getBase() + "_" + delist(dataPulls) + "_" + str(i) + ".csv"
          if type(dataStat).__name__ != 'dict':
            assert(False)
            continue
          dataPairs = []
          for key in sorted(dataStat.keys()):
            value = dataStat[key]
            if np.isscalar(value):
              dataPairs.append((key, value))
          extraCsv = Files.returnInstance('CSV',self)
          extraCsv.initialize(newFileName,self.messageHandler)
          extraCsv.open("w")
          extraCsv.write(",".join(['"' + str(x[0]) + '"' for x in dataPairs]))
          extraCsv.write("\n")
          extraCsv.write(",".join([str(x[1]) for x in dataPairs]))
          extraCsv.write("\n")
          extraCsv.close()
        utils.printCsv(csv)

  def __processData(self, data, methodInfo):
    """
      Method to process the computed data
      @ In, data, np.array, the data to process
      @ In, methodInfo, dict, the info about which processing method needs to be used
      @ Out, ret, dict, the processed data
    """
    ret = {}
    if hasattr(data,'tolist'):
      sortedData = data.tolist()
    else:
      sortedData = list(data)
    sortedData.sort()
    low = sortedData[0]
    high = sortedData[-1]
    dataRange = high - low
    ret['low'] = low
    ret['high'] = high
    if not 'binMethod' in methodInfo:
      numBins = methodInfo.get("numBins", 10)
    else:
      binMethod = methodInfo['binMethod']
      dataN = len(sortedData)
      if binMethod == 'square-root':
        numBins = int(math.ceil(math.sqrt(dataN)))
      elif binMethod == 'sturges':
        numBins = int(math.ceil(mathUtils.log2(dataN) + 1))
      else:
        self.raiseADebug("Unknown binMethod " + binMethod, 'ExceptedError')
        numBins = 5
    ret['numBins'] = numBins
    kind = methodInfo.get("kind", "uniformBins")
    if kind == "uniformBins":
      bins = [low + x * dataRange / numBins for x in range(1, numBins)]
      ret['minBinSize'] = dataRange / numBins
    elif kind == "equalProbability":
      stride = len(sortedData) // numBins
      bins = [sortedData[x] for x in range(stride - 1, len(sortedData) - stride + 1, stride)]
      if len(bins) > 1:
        ret['minBinSize'] = min(map(lambda x, y: x - y, bins[1:], bins[:-1]))
      else:
        ret['minBinSize'] = dataRange
    counts = mathUtils.countBins(sortedData, bins)
    ret['bins'] = bins
    ret['counts'] = counts
    ret.update(mathUtils.calculateStats(sortedData))
    skewness = ret["skewness"]
    delta = math.sqrt((math.pi / 2.0) * (abs(skewness) ** (2.0 / 3.0)) /
                      (abs(skewness) ** (2.0 / 3.0) + ((4.0 - math.pi) / 2.0) ** (2.0 / 3.0)))
    delta = math.copysign(delta, skewness)
    alpha = delta / math.sqrt(1.0 - delta ** 2)
    variance = ret["sampleVariance"]
    omega = variance / (1.0 - 2 * delta ** 2 / math.pi)
    mean = ret['mean']
    xi = mean - omega * delta * math.sqrt(2.0 / math.pi)
    ret['alpha'] = alpha
    ret['omega'] = omega
    ret['xi'] = xi
    return ret
#
#
#

class InterfacedPostProcessor(BasePostProcessor):
  """
    This class allows to interface a general-purpose post-processor created ad-hoc by the user.
    While the ExternalPostProcessor is designed for analysis-dependent cases, the InterfacedPostProcessor is designed more generic cases
    The InterfacedPostProcessor parses (see PostProcessorInterfaces.py) and uses only the functions contained in the raven/framework/PostProcessorFunctions folder
    The base class for the InterfacedPostProcessor that the user has to inherit to develop its own InterfacedPostProcessor is specified
    in PostProcessorInterfaceBase.py
  """

  PostProcessorInterfaces = importlib.import_module("PostProcessorInterfaces")

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self, messageHandler)
    self.methodToRun = None

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the Interfaced Post-processor
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    for child in xmlNode:
      if child.tag == 'method': self.methodToRun = child.text
    self.postProcessor = InterfacedPostProcessor.PostProcessorInterfaces.returnPostProcessorInterface(self.methodToRun,self)
    if not isinstance(self.postProcessor,PostProcessorInterfaceBase):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +' : not correctly coded; it must inherit the PostProcessorInterfaceBase class')

    self.postProcessor.initialize()
    self.postProcessor.readMoreXML(xmlNode)
    if self.postProcessor.inputFormat not in set(['HistorySet','PointSet']):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +' : self.inputFormat not correctly initialized')
    if self.postProcessor.outputFormat not in set(['HistorySet','PointSet']):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +' : self.outputFormat not correctly initialized')


  def run(self, inputIn):
    """
      This method executes the interfaced  post-processor action.
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDic, dict, dict containing the post-processed results
    """
    if self.postProcessor.inputFormat not in set(['HistorySet','PointSet']):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +' : self.inputFormat not correctly initialized')
    if self.postProcessor.outputFormat not in set(['HistorySet','PointSet']):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +' : self.outputFormat not correctly initialized')
    inputDic= self.inputToInternal(inputIn)

    outputDic = self.postProcessor.run(inputDic)
    if self.postProcessor.checkGeneratedDicts(outputDic):
      return outputDic
    else:
      self.raiseAnError(RuntimeError,'InterfacedPostProcessor Post-Processor '+ self.name +' : function has generated a not valid output dictionary')

  def _inverse(self, inputIn):
    outputDic = self.postProcessor._inverse(inputIn)
    return outputDic

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1:
      self.raiseAnError(RuntimeError, ' No available Output to collect (Run probably is not finished yet)')
    evaluation = finishedJob.getEvaluation()[1]

    exportDict = {'inputSpaceParams':evaluation['data']['input'],'outputSpaceParams':evaluation['data']['output'],'metadata':evaluation['metadata']}

    listInputParms   = output.getParaKeys('inputs')
    listOutputParams = output.getParaKeys('outputs')

    if output.type == 'HistorySet':
      for hist in exportDict['inputSpaceParams']:
        if type(exportDict['inputSpaceParams'].values()[0]).__name__ == "dict":
          for key in listInputParms:
            output.updateInputValue(key,exportDict['inputSpaceParams'][hist][str(key)])
          for key in listOutputParams:
            output.updateOutputValue(key,exportDict['outputSpaceParams'][hist][str(key)])
        else:
          for key in exportDict['inputSpaceParams']:
            if key in output.getParaKeys('inputs'):
              output.updateInputValue(key,exportDict['inputSpaceParams'][key])
          for key in exportDict['outputSpaceParams']:
            if key in output.getParaKeys('outputs'):
              output.updateOutputValue(key,exportDict['outputSpaceParams'][str(key)])
      for key in exportDict['metadata']:
        output.updateMetadata(key,exportDict['metadata'][key])
    else:   # output.type == 'PointSet':
      for key in exportDict['inputSpaceParams']:
        if key in output.getParaKeys('inputs'):
          for value in exportDict['inputSpaceParams'][key]:
            output.updateInputValue(str(key),value)
      for key in exportDict['outputSpaceParams']:
        if str(key) in output.getParaKeys('outputs'):
          for value in exportDict['outputSpaceParams'][key]:
            output.updateOutputValue(str(key),value)
      for key in exportDict['metadata']:
        output.updateMetadata(key,exportDict['metadata'][key])


  def inputToInternal(self,input):
    """
      Function to convert the received input into a format this object can
      understand
      @ In, input, list, list of dataObjects handed to the post-processor
      @ Out, inputDict, list, list of dictionaries this object can process
    """
    inputDict = []
    for inp in input:
      if type(inp) == dict:
        return [inp]
      else:
        inputDictTemp = {'data':{}, 'metadata':{}}
        inputDictTemp['data']['input']  = copy.deepcopy(inp.getInpParametersValues())
        inputDictTemp['data']['output'] = copy.deepcopy(inp.getOutParametersValues())
        inputDictTemp['metadata']       = copy.deepcopy(inp.getAllMetadata())
        inputDictTemp['name'] = inp.whoAreYou()['Name']
        inputDict.append(inputDictTemp)
    return inputDict

#
#
#

class ImportanceRankInput(InputData.ParameterInput):
  """
    class for reading in the ImportanceRank block
  """

ImportanceRankInput.createClass("PostProcessor", False, baseNode=ModelInput)
WhatInput = InputData.parameterInputFactory("what", contentType=InputData.StringType)
ImportanceRankInput.addSub(WhatInput)
VariablesInput = InputData.parameterInputFactory("variables", contentType=InputData.StringType)
DimensionsInput = InputData.parameterInputFactory("dimensions", contentType=InputData.StringType)
ManifestInput = InputData.parameterInputFactory("manifest", contentType=InputData.StringType)
ManifestInput.addSub(VariablesInput)
ManifestInput.addSub(DimensionsInput)
LatentInput = InputData.parameterInputFactory("latent", contentType=InputData.StringType)
LatentInput.addSub(VariablesInput)
LatentInput.addSub(DimensionsInput)
FeaturesInput = InputData.parameterInputFactory("features", contentType=InputData.StringType)
FeaturesInput.addSub(ManifestInput)
FeaturesInput.addSub(LatentInput)
ImportanceRankInput.addSub(FeaturesInput)
TargetsInput = InputData.parameterInputFactory("targets", contentType=InputData.StringType)
ImportanceRankInput.addSub(TargetsInput)
#DimensionsInput = InputData.parameterInputFactory("dimensions", contentType=InputData.StringType)
#ImportanceRankInput.addSub(DimensionsInput)
MVNDistributionInput = InputData.parameterInputFactory("mvnDistribution", contentType=InputData.StringType)
ImportanceRankInput.addSub(MVNDistributionInput)
PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputData.StringType)
ImportanceRankInput.addSub(PivotParameterInput)

#
class ImportanceRank(BasePostProcessor):
  """
    ImportantRank class. It computes the important rank for given input parameters
    1. The importance of input parameters can be ranked via their sensitivies (SI: sensitivity index)
    2. The importance of input parameters can be ranked via their sensitivies and covariances (II: importance index)
    3. The importance of input directions based principal component analysis of inputs covariances (PCA index)
    3. CSI: Cumulative sensitive index (added in the future)
    4. CII: Cumulative importance index (added in the future)
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self, messageHandler)
    self.targets = []
    self.features = []
    self.latent = []
    self.latentDim = []
    self.manifest = []
    self.manifestDim = []
    self.dimensions = []
    self.mvnDistribution = None
    self.acceptedMetric = ['sensitivityindex','importanceindex','pcaindex','transformation','inversetransformation','manifestsensitivity']
    self.all = ['sensitivityindex','importanceindex','pcaindex']
    self.statAcceptedMetric = ['pcaindex','transformation','inversetransformation']
    self.what = self.acceptedMetric # what needs to be computed, default is all
    self.printTag = 'POSTPROCESSOR IMPORTANTANCE RANK'
    self.requiredAssObject = (True,(['Distributions'],[-1]))
    self.transformation = False
    self.latentSen = False
    self.reconstructSen = False
    self.pivotParameter = None # time-dependent pivot parameter
    self.dynamic        = False # is it time-dependent?

  def _localWhatDoINeed(self):
    """
      This method is local mirror of the general whatDoINeed method
      It is implemented by this postprocessor that need to request special objects
      @ In, None
      @ Out, needDict, dict, list of objects needed
    """
    needDict = {'Distributions':[]}
    needDict['Distributions'].append((None,self.mvnDistribution))
    return needDict

  def _localGenerateAssembler(self,initDict):
    """
      see generateAssembler method in Assembler
      @ In, initDict, dict, dictionary ({'mainClassName':{'specializedObjectName':ObjectInstance}})
      @ Out, None
    """
    distName = self.mvnDistribution
    self.mvnDistribution = initDict['Distributions'][distName]

  def _localReadMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs
      @ In, xmlNode, xml.etree.ElementTree Element Objects, the xml element node that will be checked against the available options specific to this Sampler
      @ Out, None
    """
    paramInput = ImportanceRankInput()
    paramInput.parseNode(xmlNode)
    for child in paramInput.subparts:
      if child.getName() == 'what':
        what = child.value.strip()
        if what.lower() == 'all': self.what = self.all
        else:
          requestMetric = list(var.strip() for var in what.split(','))
          toCalculate = []
          for metric in requestMetric:
            if metric.lower() == 'all':
              toCalculate.extend(self.all)
            elif metric.lower() in self.acceptedMetric:
              if metric.lower() not in toCalculate:
                toCalculate.append(metric.lower())
              else:
                self.raiseAWarning('Duplicate calculations',metric,'are removed from XML node <what> in',self.printTag)
            else:
              self.raiseAnError(IOError, self.printTag,'asked unknown operation', metric, '. Available',str(self.acceptedMetric))
          self.what = toCalculate
      elif child.getName() == 'targets':
        self.targets = list(inp.strip() for inp in child.value.strip().split(','))
      elif child.getName() == 'features':
        for subNode in child.subparts:
          if subNode.getName() == 'manifest':
            for subSubNode in subNode.subparts:
              if subSubNode.getName() == 'variables':
                self.manifest = list(inp.strip() for inp in subSubNode.value.strip().split(','))
                self.features.extend(self.manifest)
              elif subSubNode.getName() == 'dimensions':
                self.manifestDim = list(int(inp.strip()) for inp in subSubNode.value.strip().split(','))
              else:
                self.raiseAnError(IOError, 'Unrecognized xml node name:',subSubNode.getName(),'in',self.printTag)
          if subNode.getName() == 'latent':
            self.latentSen = True
            for subSubNode in subNode.subparts:
              if subSubNode.getName() == 'variables':
                self.latent = list(inp.strip() for inp in subSubNode.value.strip().split(','))
                self.features.extend(self.latent)
              elif subSubNode.getName() == 'dimensions':
                self.latentDim = list(int(inp.strip()) for inp in subSubNode.value.strip().split(','))
              else:
                self.raiseAnError(IOError, 'Unrecognized xml node name:',subSubNode.getName(),'in',self.printTag)
      elif child.getName() == 'mvnDistribution':
        self.mvnDistribution = child.value.strip()
      elif child.getName() == "pivotParameter": self.pivotParameter = child.value
      else:
        self.raiseAnError(IOError, 'Unrecognized xml node name: ' + child.getName() + '!')
    if not self.latentDim and len(self.latent) != 0:
      self.latentDim = range(1,len(self.latent)+1)
      self.raiseAWarning('The dimensions for given latent variables: ' + str(self.latent) + ' is not provided! Default dimensions will be used: ' + str(self.latentDim) + ' in ' + self.printTag)
    if not self.manifestDim and len(self.manifest) !=0:
      self.manifestDim = range(1,len(self.manifest)+1)
      self.raiseAWarning('The dimensions for given latent variables: ' + str(self.manifest) + ' is not provided! Default dimensions will be used: ' + str(self.manifestDim) + ' in ' + self.printTag)
    if not self.features:
      self.raiseAnError(IOError, 'No variables provided for XML node: features in',self.printTag)
    if not self.targets:
      self.raiseAnError(IOError, 'No variables provided for XML node: targets in', self.printTag)
    if len(self.latent) !=0 and len(self.manifest) !=0:
      self.reconstructSen = True
      self.transformation = True

  def _localPrintXML(self,outFile,options=None,pivotVal=None):
    """
      Adds requested entries to XML node.
      @ In, outFile, Files.StaticXMLOutput, file to which entries will be printed
      @ In, options, dict, optional, list of requests and options
        May include: 'what': comma-separated string list, the qualities to print out
      @ In, pivotVal, float, value of the pivot parameter, i.e. time, burnup, ...
      @ Out, None
    """
    #build tree
    for what in options.keys():
      if what.lower() in self.statAcceptedMetric: continue
      if what == 'manifestSensitivity': continue
      for target in options[what].keys():
        valueDict = OrderedDict()
        for var,index,dim in options[what][target]:
          valueDict[var] = index
        outFile.addVector(target,what,valueDict,pivotVal=pivotVal)
    if 'manifestSensitivity' in options.keys():
      what = 'manifestSensitivity'
      for target in options[what].keys():
        outFile.addScalar(target,'type',self.mvnDistribution.covarianceType,pivotVal=pivotVal)
        valueDict = OrderedDict()
        for var,index,dim in options[what][target]:
          valueDict[var] = index
        outFile.addVector(target,what,valueDict,pivotVal=pivotVal)

  def _localPrintPCAInformation(self,outFile,options=None,pivotVal=None):
    """
      Adds requested entries to XML node.
      @ In, outFile, Files.StaticXMLOutput, file to which entries will be printed
      @ In, options, dict, optional, list of requests and options
        May include: 'what': comma-separated string list, the qualities to print out
      @ In, pivotVal, float, value of the pivot parameter, i.e. time, burnup, ...
      @ Out, None
    """
    # output variables and dimensions
    latentDict = OrderedDict()
    if self.latentSen:
      for index,var in enumerate(self.latent):
        latentDict[var] = self.latentDim[index]
      outFile.addVector('dimensions','latent',latentDict,pivotVal=pivotVal)
    manifestDict = OrderedDict()
    if len(self.manifest) > 0:
      for index,var in enumerate(self.manifest):
        manifestDict[var] = self.manifestDim[index]
      outFile.addVector('dimensions','manifest',manifestDict,pivotVal=pivotVal)
    #pca index is a feature only of target, not with respect to anything else
    if 'pcaIndex' in options.keys():
      pca = options['pcaIndex']
      for var,index,dim in pca:
        outFile.addScalar('pcaIndex',var,index,pivotVal=pivotVal)
    if 'transformation' in options.keys():
      what = 'transformation'
      outFile.addScalar(what,'type',self.mvnDistribution.covarianceType,pivotVal=pivotVal)
      for target in options[what].keys():
        valueDict = OrderedDict()
        for var, index, dim in options[what][target]:
          valueDict[var] = index
        outFile.addVector(what,target,valueDict,pivotVal=pivotVal)
    if 'inverseTransformation' in options.keys():
      what = 'inverseTransformation'
      outFile.addScalar(what,'type',self.mvnDistribution.covarianceType,pivotVal=pivotVal)
      for target in options[what].keys():
        valueDict = OrderedDict()
        for var, index, dim in options[what][target]:
          valueDict[var] = index
        outFile.addVector(what,target,valueDict,pivotVal=pivotVal)

  def collectOutput(self,finishedJob, output):
    """
      Function to place all of the computed data into the output object, (Files or DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1: self.raiseAnError(RuntimeError, ' No available output to collect (Run probably is not finished yet) via',self.printTag)
    outputDict = finishedJob.getEvaluation()[-1]
    # Output to file
    if isinstance(output, Files.File):
      availExtens = ['xml','csv']
      outputExtension = output.getExt().lower()
      if outputExtension not in availExtens:
        self.raiseAWarning('Output extension you input is ' + outputExtension)
        self.raiseAWarning('Available are ' + str(availExtens) + '. Converting extension to ' + str(availExtens[0]) + '!')
        outputExtensions = availExtens[0]
        output.setExtension(outputExtensions)
      output.setPath(self.__workingDir)
      self.raiseADebug('Dumping output in file named ' + output.getAbsFile())
      output.open('w')
      if outputExtension == 'csv':
        self._writeCSV(output,outputDict)
      else:
        self._writeXML(output,outputDict)
    # Output to DataObjects
    elif output.type in ['PointSet','HistorySet']:
      self.raiseADebug('Dumping output in data object named ' + output.name)
      self._writeDataObject(output,outputDict)
    elif output.type == 'HDF5' : self.raiseAWarning('Output type ' + str(output.type) + ' not yet implemented. Skip it !!!!!')
    else: self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')

  def _writeCSV(self,output,outputDictionary):
    """
      Defines the method for writing the post-processor to a .csv file
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary stores importance ranking outputs
      @ Out, None
    """
    separator = ','
    if self.dynamic:
      output.write('Importance Rank' + separator + 'Pivot Parameter' + separator + self.pivotParameter + os.linesep)
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for step, outputDict in enumerate(outputResults):
      if self.dynamic: output.write('Pivot Value'+separator+str(outputDictionary.keys()[step])+os.linesep)
      #only output 'pcaindex','transformation','inversetransformation' for the first step.
      if step == 0:
        for what in outputDict.keys():
          if what.lower() in self.statAcceptedMetric:
            self.raiseADebug('Writing parameter rank for metric ' + what)
            if what.lower() == 'pcaindex':
              output.write('pcaIndex,' + '\n')
              output.write('Parameters'  + ''.join([separator + str(item[0])  for item in outputDict[what]]) + os.linesep)
              output.write(what + ''.join([separator + '%.8E' % item[1] for item in outputDict[what]]) + os.linesep)
              output.write(os.linesep)
            else:
              for target in outputDict[what].keys():
                output.write('Target,' + target + '\n')
                output.write('Parameters'  + ''.join([separator + str(item[0])  for item in outputDict[what][target]]) + os.linesep)
                output.write(what + ''.join([separator + '%.8E' % item[1] for item in outputDict[what][target]]) + os.linesep)
              output.write(os.linesep)
      for what in outputDict.keys():
        if what.lower() in self.statAcceptedMetric: continue
        if what.lower() in self.acceptedMetric:
          self.raiseADebug('Writing parameter rank for metric ' + what)
          for target in outputDict[what].keys():
            output.write('Target,' + target + '\n')
            output.write('Parameters'  + ''.join([separator + str(item[0])  for item in outputDict[what][target]]) + os.linesep)
            output.write(what + ''.join([separator + '%.8E' % item[1] for item in outputDict[what][target]]) + os.linesep)
          output.write(os.linesep)
    output.close()

  def _writeXML(self,output,outputDictionary):
    """
      Defines the method for writing the post-processor to a .csv file
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary stores importance ranking outputs
      @ Out, None
    """
    if output.isOpen(): output.close()
    if self.dynamic:
      outFile = Files.returnInstance('DynamicXMLOutput',self)
    else:
      outFile = Files.returnInstance('StaticXMLOutput',self)
    outFile.initialize(output.getFilename(),self.messageHandler,path=output.getPath())
    outFile.newTree('ImportanceRankPP',pivotParam=self.pivotParameter)
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for step, outputDict in enumerate(outputResults):
      pivotVal = outputDictionary.keys()[step]
      self._localPrintXML(outFile,outputDict,pivotVal)
      if step == 0:
        self._localPrintPCAInformation(outFile,outputDict,pivotVal)
    outFile.writeFile()
    self.raiseAMessage('ImportanceRank XML printed to "'+output.getFilename()+'"!')

  def _writeDataObject(self,output,outputDictionary):
    """
      Defines the method for writing the post-processor to a .csv file
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary stores importance ranking outputs
      @ Out, None
    """
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for step, outputDict in enumerate(outputResults):
      if step == 0:
        for what in outputDict.keys():
          if what.lower() not in self.statAcceptedMetric:
            continue
          elif what.lower() == 'pcaindex':
            self.raiseADebug('Dumping ' + what + '. Metadata name = ' + what)
            output.updateMetadata(what,outputDict[what])
          else:
            for target in outputDict[what].keys():
              self.raiseADebug('Dumping ' + target + '-' + what + '. Metadata name = ' + target + '-' + what + '. Targets stored in ' +  target + '-'  + what)
              output.updateMetadata(target + '-' + what, outputDict[what][target])
      appendix = '-'+self.pivotParameter+'-'+str(outputDictionary.keys()[step]) if self.dynamic else ''
      for what in outputDict.keys():
        if what.lower() in self.statAcceptedMetric: continue
        if what.lower() in self.acceptedMetric:
          for target in outputDict[what].keys():
            self.raiseADebug('Dumping ' + target + '-' + what + '. Metadata name = ' + target + '-' + what + '. Targets stored in ' +  target + '-'  + what)
            output.updateMetadata(target + '-'  + what+appendix, outputDict[what][target])

  def initialize(self, runInfo, inputs, initDict) :
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
    """
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = runInfo['WorkingDir']

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputDict, dictionary of the converted data
    """
    if type(currentInp) == list  : currentInput = currentInp[-1]
    else                         : currentInput = currentInp
    if type(currentInput) == dict:
      if 'targets' not in currentInput.keys() and 'timeDepData' not in currentInput.keys(): self.raiseAnError(IOError, 'Did not find targets or timeDepData in input dictionary')
      return currentInput

    if hasattr(currentInput,'type'):
      inType = currentInput.type
    else:
      if type(currentInput).__name__ == 'list'    : inType = 'list'
      else: self.raiseAnError(IOError, self, 'ImportanceRank postprocessor accepts Files, HDF5, PointSet, DataObject(s) only! Got ' + str(type(currentInput)))
    if inType not in ['HDF5', 'PointSet','HistorySet', 'list'] and not isinstance(inType,Files.File):
      self.raiseAnError(IOError, self, 'ImportanceRank postprocessor accepts Files, HDF5, HistorySet, PointSet, DataObject(s) only! Got ' + str(inType) + '!!!!')
    # get input from the external csv file
    if isinstance(inType,Files.File):
      if currentInput.subtype == 'csv': pass # to be implemented
    # get input from PointSet DataObject
    if inType in ['PointSet']:
      inputDict = {'targets':{}, 'metadata':{}, 'features':{}}
      for feat in self.features:
        if feat in currentInput.getParaKeys('input'):
          inputDict['features'][feat] = currentInput.getParam('input', feat)
        else:
          self.raiseAnError(IOError,'Parameter ' + str(feat) + ' is listed ImportanceRank postprocessor features, but not found in the provided input!')
      for targetP in self.targets:
        if targetP in currentInput.getParaKeys('output'):
          inputDict['targets'][targetP] = currentInput.getParam('output', targetP)
        else:
          self.raiseAnError(IOError,'Parameter ' + str(targetP) + ' is listed ImportanceRank postprocessor targets, but not found in the provided output!')
      inputDict['metadata'] = currentInput.getAllMetadata()
    # get input from HistorySet DataObject
    if inType in ['HistorySet']:
      if self.pivotParameter is None: self.raiseAnError(IOError, self, 'Time-dependent importance ranking is requested (HistorySet) but no pivotParameter got inputted!')
      inputs  = currentInput.getParametersValues('inputs',nodeId = 'ending')
      outputs = currentInput.getParametersValues('outputs',nodeId = 'ending')
      numSteps = len(outputs.values()[0].values()[0])
      self.dynamic = True
      if self.pivotParameter not in currentInput.getParaKeys('output'):
        self.raiseAnError(IOError, self, 'Pivot parameter ' + self.pivotParameter + ' has not been found in output space of data object '+currentInput.name)
      pivotParameter = []
      for step in range(len(outputs.values()[0][self.pivotParameter])):
        currentSnapShot = [outputs[i][self.pivotParameter][step] for i in outputs.keys()]
        if len(set(currentSnapShot)) > 1: self.raiseAnError(IOError, self, 'Histories are not syncronized! Please, pre-process the data using Interfaced PostProcessor HistorySetSync!')
        pivotParameter.append(currentSnapShot[-1])
      inputDict = {'timeDepData':OrderedDict.fromkeys(pivotParameter,None)}
      for step in range(numSteps):
        inputDict['timeDepData'][pivotParameter[step]] = {'targets':{},'features':{}}
        for targetP in self.targets:
          if targetP in currentInput.getParaKeys('output') :
            inputDict['timeDepData'][pivotParameter[step]]['targets'][targetP] = np.asarray([outputs[i][targetP][step] for i in outputs.keys()])
          else:
            self.raiseAnError(IOError, self, 'Target ' + targetP + ' has not been found in data object '+currentInput.name)
        for feat in self.features:
          if feat in currentInput.getParaKeys('input'):
            inputDict['timeDepData'][pivotParameter[step]]['features'][feat] = np.asarray([inputs[i][feat][-1] for i in inputs.keys()])
          else:
            self.raiseAnError(IOError, self, 'Feature ' + feat + ' has not been found in data object '+currentInput.name)
        inputDict['timeDepData'][pivotParameter[step]]['metadata'] = currentInput.getAllMetadata()

    # get input from HDF5 Database
    if inType == 'HDF5': pass  # to be implemented

    return inputDict

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    inputDict = self.inputToInternal(inputIn)
    if not self.dynamic: outputDict = self.__runLocal(inputDict)
    else:
      # time dependent (actually pivot-dependent)
      outputDict = OrderedDict()
      for pivotParamValue in inputDict['timeDepData'].keys():
        outputDict[pivotParamValue] = self.__runLocal(inputDict['timeDepData'][pivotParamValue])
    return outputDict

  def __runLocal(self, inputDict):
    """
      This method executes the postprocessor action.
      @ In, inputDict, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, dictionary containing the evaluated data
    """
    outputDict = {}
    senCoeffDict = {}
    senWeightDict = {}
    # compute sensitivities of targets with respect to features
    featValues = []
    # compute importance rank
    if self.latentSen:
      for feat in self.latent:
        featValues.append(inputDict['features'][feat])
      feats = self.latent
      self.dimensions = self.latentDim
    else:
      for feat in self.manifest:
        featValues.append(inputDict['features'][feat])
      feats = self.manifest
      self.dimensions = self.manifestDim
    sampledFeatMatrix = np.atleast_2d(np.asarray(featValues)).T
    for target in self.targets:
      featCoeffs = LinearRegression().fit(sampledFeatMatrix, inputDict['targets'][target]).coef_
      featWeights = abs(featCoeffs)/np.sum(abs(featCoeffs))
      senWeightDict[target] = list(zip(feats,featWeights,self.dimensions))
      senCoeffDict[target] = featCoeffs
    for what in self.what:
      if what.lower() == 'sensitivityindex':
        what = 'sensitivityIndex'
        if what not in outputDict.keys(): outputDict[what] = {}
        for target in self.targets:
          entries = senWeightDict[target]
          entries.sort(key=lambda x: x[1],reverse=True)
          outputDict[what][target] = entries
      if what.lower() == 'importanceindex':
        what = 'importanceIndex'
        if what not in outputDict.keys(): outputDict[what] = {}
        for target in self.targets:
          featCoeffs = senCoeffDict[target]
          featWeights = []
          if not self.latentSen:
            for index,feat in enumerate(self.manifest):
              totDim = self.mvnDistribution.dimension
              covIndex = totDim * (self.dimensions[index] - 1) + self.dimensions[index] - 1
              if self.mvnDistribution.covarianceType == 'abs':
                covTarget = featCoeffs[index] * self.mvnDistribution.covariance[covIndex] * featCoeffs[index]
              else:
                covFeature = self.mvnDistribution.covariance[covIndex]*self.mvnDistribution.mu[self.dimensions[index]-1]**2
                covTarget = featCoeffs[index] * covFeature * featCoeffs[index]
              featWeights.append(covTarget)
            featWeights = featWeights/np.sum(featWeights)
            entries = list(zip(self.manifest,featWeights,self.dimensions))
            entries.sort(key=lambda x: x[1],reverse=True)
            outputDict[what][target] = entries
          # if the features type is 'latent', since latentVariables are used to compute the sensitivities
          # the covariance for latentVariances are identity matrix
          else:
            entries = senWeightDict[target]
            entries.sort(key=lambda x: x[1],reverse=True)
            outputDict[what][target] = entries
      #calculate PCA index
      if what.lower() == 'pcaindex':
        if not self.latentSen:
          self.raiseAWarning('pcaIndex can be not requested because no latent variable is provided!')
        else:
          what = 'pcaIndex'
          if what not in outputDict.keys(): outputDict[what] = {}
          index = [dim-1 for dim in self.dimensions]
          singularValues = self.mvnDistribution.returnSingularValues(index)
          singularValues = list(singularValues/np.sum(singularValues))
          entries = list(zip(self.latent,singularValues,self.dimensions))
          entries.sort(key=lambda x: x[1],reverse=True)
          outputDict[what] = entries

      if what.lower() == 'transformation':
        if self.transformation:
          what = 'transformation'
          if what not in outputDict.keys(): outputDict[what] = {}
          index = [dim-1 for dim in self.latentDim]
          manifestIndex = [dim-1 for dim in self.manifestDim]
          transformMatrix = self.mvnDistribution.transformationMatrix(index)
          for ind,var in enumerate(self.manifest):
            entries = list(zip(self.latent,transformMatrix[manifestIndex[ind]],self.latentDim))
            outputDict[what][var] = entries
        else:
          self.raiseAnError(IOError,'Unable to output the transformation matrix, please provide both "manifest" and "latent" variables in XML node "features" in',self.printTag)
      if what.lower() == 'inversetransformation':
        if self.transformation:
          what = 'inverseTransformation'
          if what not in outputDict.keys(): outputDict[what] = {}
          index = [dim-1 for dim in self.latentDim]
          manifestIndex = [dim-1 for dim in self.manifestDim]
          inverseTransformationMatrix = self.mvnDistribution.inverseTransformationMatrix(manifestIndex)
          for ind,var in enumerate(self.latent):
            entries = list(zip(self.manifest,inverseTransformationMatrix[index[ind]],self.manifestDim))
            outputDict[what][var] = entries
        else:
          self.raiseAnError(IOError,'Unable to output the inverse transformation matrix, please provide both "manifest" and "latent" variables in XML node "features" in', self.printTag)

      if what.lower() == 'manifestsensitivity':
        if self.reconstructSen:
          what = 'manifestSensitivity'
          if what not in outputDict.keys(): outputDict[what] = {}
          # compute the inverse transformation matrix
          index = [dim-1 for dim in self.latentDim]
          manifestIndex = [dim-1 for dim in self.manifestDim]
          inverseTransformationMatrix = self.mvnDistribution.inverseTransformationMatrix(manifestIndex)
          inverseTransformationMatrix = inverseTransformationMatrix[index]
          # recompute the sensitivities for manifest variables
          for target in self.targets:
            latentSen = np.asarray(senCoeffDict[target])
            if self.mvnDistribution.covarianceType == 'abs':
              manifestSen = list(np.dot(latentSen,inverseTransformationMatrix))
            else:
              manifestSen = list(np.dot(latentSen,inverseTransformationMatrix)/inputDict['targets'][target])
            entries = list(zip(self.manifest,manifestSen,self.manifestDim))
            entries.sort(key=lambda x: abs(x[1]),reverse=True)
            outputDict[what][target] = entries
        elif self.latentSen:
          self.raiseAnError(IOError, 'Unable to reconstruct the sensitivities for manifest variables, this is because no manifest variable is provided in',self.printTag)
        else:
          self.raiseAWarning('No latent variables, and there is no need to reconstruct the sensitivities for manifest variables!')

      # To be implemented
      #if what == 'CumulativeSenitivityIndex':
      #  self.raiseAnError(NotImplementedError,'CumulativeSensitivityIndex is not yet implemented for ' + self.printTag)
      #if what == 'CumulativeImportanceIndex':
      #  self.raiseAnError(NotImplementedError,'CumulativeImportanceIndex is not yet implemented for ' + self.printTag)

    return outputDict

#
#

#class BasicStatisticsInput(InputData.ParameterInput):
#  """
#    Class for reading the Basic Statistics block
#  """

#BasicStatisticsInput.createClass("PostProcessor", False, baseNode=ModelInput)
#BasicStatisticsInput.addSub(WhatInput)
#BiasedInput = InputData.parameterInputFactory("biased", contentType=InputData.StringType) #bool
#BasicStatisticsInput.addSub(BiasedInput)
#ParameterInput = InputData.parameterInputFactory("parameters", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(ParameterInput)
#MethodsToRunInput = InputData.parameterInputFactory("methodsToRun", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(MethodsToRunInput)
#FunctionInput = InputData.parameterInputFactory("Function", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(FunctionInput)
#PivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputData.StringType)
#BasicStatisticsInput.addSub(PivotParameterInput)

#
class BasicStatistics(BasePostProcessor):
  """
    BasicStatistics filter class. It computes all the most popular statistics
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self, messageHandler)
    self.parameters = {}  # parameters dictionary (they are basically stored into a dictionary identified by tag "targets"
    self.scalarVals = ['expectedValue',
                       'minimum',
                       'maximum',
                       'median',
                       'variance',
                       'sigma',
                       'percentile',
                       'variationCoefficient',
                       'skewness',
                       'kurtosis',
                       'samples']
    self.vectorVals = ['sensitivity',
                       'covariance',
                       'pearson',
                       'NormalizedSensitivity',
                       'VarianceDependentSensitivity']
    self.acceptedCalcParam = self.scalarVals + self.vectorVals
    self.what = self.acceptedCalcParam  # what needs to be computed... default...all
    self.methodsToRun = []  # if a function is present, its outcome name is here stored... if it matches one of the known outcomes, the pp is going to use the function to compute it
    self.externalFunction = []
    self.printTag = 'POSTPROCESSOR BASIC STATISTIC'
    self.addAssemblerObject('Function','-1', True)
    self.biased = False # biased statistics?
    self.pivotParameter = None # time-dependent statistics pivot parameter
    self.dynamic        = False # is it time-dependent?

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInp, object, an object that needs to be converted
      @ Out, inputDict, dict, dictionary of the converted data
    """
    # each post processor knows how to handle the coming inputs. The BasicStatistics postprocessor accept all the input type (files (csv only), hdf5 and datas
    self.dynamic = False
    currentInput = currentInp [-1] if type(currentInp) == list else currentInp
    if len(currentInput) == 0: self.raiseAnError(IOError, "In post-processor " +self.name+" the input "+currentInput.name+" is empty.")

    if type(currentInput).__name__ =='dict':
      if 'targets' not in currentInput.keys() and 'timeDepData' not in currentInput.keys(): self.raiseAnError(IOError, 'Did not find targets or timeDepData in input dictionary')
      return currentInput
    if currentInput.type not in ['PointSet','HistorySet']: self.raiseAnError(IOError, self, 'BasicStatistics postprocessor accepts PointSet and HistorySet only! Got ' + currentInput.type)
    if currentInput.type in ['PointSet']:
      inputDict = {'targets':{},'metadata':currentInput.getAllMetadata()}
      for targetP in self.parameters['targets']:
        if   targetP in currentInput.getParaKeys('input') : inputDict['targets'][targetP] = currentInput.getParam('input' , targetP, nodeId = 'ending')
        elif targetP in currentInput.getParaKeys('output'): inputDict['targets'][targetP] = currentInput.getParam('output', targetP, nodeId = 'ending')
        else: self.raiseAnError(IOError, self, 'Target ' + targetP + ' has not been found in data object '+currentInput.name)
    else:
      if self.pivotParameter is None: self.raiseAnError(IOError, self, 'Time-dependent statistics is requested (HistorySet) but no pivotParameter got inputted!')
      inputs, outputs  = currentInput.getParametersValues('inputs',nodeId = 'ending'), currentInput.getParametersValues('outputs',nodeId = 'ending')
      nTs, self.dynamic = len(outputs.values()[0].values()[0]), True
      if self.pivotParameter not in currentInput.getParaKeys('output'): self.raiseAnError(IOError, self, 'Pivot parameter ' + self.pivotParameter + ' has not been found in output space of data object '+currentInput.name)
      pivotParameter =  six.next(six.itervalues(outputs))[self.pivotParameter]
      self.raiseAMessage("Starting recasting data for time-dependent statistics")
      targetInput  = []
      targetOutput = []
      for targetP in self.parameters['targets']:
        if targetP in currentInput.getParaKeys('output'):
          targetOutput.append(targetP)
        elif targetP in currentInput.getParaKeys('input'):
          targetInput.append(targetP)
        else: self.raiseAnError(IOError, self, 'Target ' + targetP + ' has not been found in data object '+currentInput.name)
      inputDict = {}
      inputDict['timeDepData'] = OrderedDict((el,defaultdict(dict)) for el in pivotParameter)
      for targetP in targetInput:
        inputValues = np.asarray([val[targetP][-1] for val in inputs.values()])
        for ts in range(nTs): inputDict['timeDepData'][pivotParameter[ts]]['targets'][targetP] = inputValues
      metadata = currentInput.getAllMetadata()
      for cnt, targetP in enumerate(targetOutput):
        outputValues = np.asarray([val[targetP] for val in outputs.values()])
        if len(outputValues.shape) != 2: self.raiseAnError(IOError, 'Histories are not syncronized! Please, pre-process the data using Interfaced PostProcessor HistorySetSync!')
        for ts in range(nTs):
          inputDict['timeDepData'][pivotParameter[ts]]['targets'][targetP] = outputValues[:,ts]
          if cnt == 0 : inputDict['timeDepData'][pivotParameter[ts]]['metadata'] = metadata
    self.raiseAMessage("Recasting performed")
    return inputDict

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the BasicStatistic pp. In here the working dir is
      grepped.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    #construct a list of all the parameters that have requested values into self.allUsedParams
    self.allUsedParams = set()
    #first collect parameters for which scalar values were requested
    for scalar in self.scalarVals:
      if scalar in self.toDo.keys():
        #special treatment of percentile since the user can specify the percents directly
        if scalar == 'percentile':
          for pct,targs in self.toDo[scalar].items():
            self.allUsedParams.update(targs)
        else:
          self.allUsedParams.update(self.toDo[scalar])
    #second collect parameters for which matrix values were requested, either as targets or features
    for vector in self.vectorVals:
      if vector in self.toDo.keys():
        for entry in self.toDo[vector]:
          self.allUsedParams.update(entry['targets'])
          self.allUsedParams.update(entry['features'])
    #for backward compatibility, compile the full list of parameters used in Basic Statistics calculations
    self.parameters['targets'] = list(self.allUsedParams)
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = runInfo['WorkingDir']

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    self.toDo = {}
    for child in xmlNode:
      tag = child.tag.strip()
      #because percentile is strange (has an attached parameter), we address it first
      if tag.startswith('percentile'):
        #get targets
        targets = set(a.strip() for a in child.text.split(','))
        #what if user didn't give any targets?
        if len(targets)<1:
          self.raiseAWarning('No targets were specified in text of <'+tag+'>!  Skipping metric...')
          continue
        #prepare storage dictionary, keys are percentiles, values are set(targets)
        if 'percentile' not in self.toDo.keys():
          self.toDo['percentile']={}
          self.parameters['percentile_map'] = {}
        if tag == 'percentile':
          floatPercentile = [float(5),float(95)]
          self.parameters['percentile_map'][floatPercentile[0]] = '5'
          self.parameters['percentile_map'][floatPercentile[1]] = '95'
        else:
          #user specified a percentage!
          splitTag = tag.split('_')
          if len(splitTag) != 2:
            self.raiseAWarning('Not able to parse "'+tag+'" to obtain percentile!  Expected "percentile_##%". Using 95% instead...')
            floatPercentile = [float(95)]
            self.parameters['percentile_map'][floatPercentile[-1]] = '95'
          else:
            floatPercentile = [utils.floatConversion(splitTag[1].replace("%",""))]
            self.parameters['percentile_map'][floatPercentile[-1]] = splitTag[1]
            if floatPercentile[0] is None:
              self.raiseAWarning('Not able to parse "'+tag+'" to obtain percentile!  Could not parse',strPercent,'as a percentile. Using 95% instead...')
              floatPercentile = [float(95)]
              self.parameters['percentile_map'][floatPercentile[-1]] = '95'
        for reqPercent in floatPercentile:
          if reqPercent in self.toDo['percentile'].keys():
            self.toDo['percentile'][reqPercent].update(targets)
          else:
            self.toDo['percentile'][reqPercent] = set(targets)
      elif tag in self.scalarVals:
        if tag in self.toDo.keys():
          self.toDo[tag].update(set(a.strip() for a in child.text.split(',')))
        else:
          self.toDo[tag] = set(a.strip() for a in child.text.split(','))
      elif tag in self.vectorVals:
        self.toDo[tag] = [] #'inputs':[],'outputs':[]}
        tnode = child.find('targets')
        if tnode is None:
          self.raiseAnError('Request for vector value <'+tag+'> requires a "targets" node, and none was found!')
        fnode = child.find('features')
        if fnode is None:
          self.raiseAnError('Request for vector value <'+tag+'> requires a "features" node, and none was found!')
        if tag in self.toDo.keys():
          # we're storing toDo[tag] as a list of dictionaries.  This is because the user might specify multiple
          #   nodes with the same metric (tag), but with different targets and features.  For instance, the user might
          #   want the sensitivity of A and B to X and Y, and the sensitivity of C to W and Z, but not the sensitivity
          #   of A to W.  If we didn't keep them separate, we could potentially waste a fair number of calculations.
          self.toDo[tag].append({'targets':set(a.strip() for a in fnode.text.split(',')),
                            'features':set(a.strip() for a in tnode.text.split(','))})
        else:
          self.toDo[tag] = [{'targets':set(a.strip() for a in fnode.text.split(',')),
                            'features':set(a.strip() for a in tnode.text.split(','))}]
      elif tag == 'all':
        #do all the metrics
        #establish targets and features
        # - as currently done, we only do the scalar metrics for the targets
        #   and features are for the matrix operations
        tnode = child.find('targets')
        if tnode is None:
          self.raiseAnError(IOError,'When using "all" node, you must specify a "targets" and a "features" node!  "targets" is missing.')
        fnode = child.find('features')
        if fnode is None:
          self.raiseAnError(IOError,'When using "all" node, you must specify a "targets" and a "features" node!  "features" is missing.')
        targets = set(a.strip() for a in tnode.text.split(','))
        features = set(a.strip() for a in fnode.text.split(','))
        for scalar in self.scalarVals:
          #percentile is a little different
          if scalar == 'percentile':
            if scalar not in self.toDo.keys():
              self.toDo[scalar] = {}
              self.parameters[scalar+'_map'] = {}
            for pct in [float(5),float(95)]:
              self.parameters['percentile_map'][pct] = str(int(pct))
              if pct in self.toDo[scalar].keys():
                self.toDo[scalar][pct].update(targets)
              else:
                self.toDo[scalar][pct] = set(targets)
          #other scalars are simple
          else:
            if scalar not in self.toDo.keys():
              self.toDo[scalar] = set()
            self.toDo[scalar].update(set(a.strip() for a in tnode.text.split(',')))
        for vector in self.vectorVals:
          if vector not in self.toDo.keys():
            self.toDo[vector] = []
          self.toDo[vector].append({'targets':set(a.strip() for a in fnode.text.split(',')),
                                 'features':set(a.strip() for a in tnode.text.split(','))})
      elif child.tag == "biased":
        if child.text.lower() in utils.stringsThatMeanTrue():
          self.biased = True
      elif child.tag == "pivotParameter":
        self.pivotParameter = child.text
      else:
        self.raiseAWarning('Unrecognized node in BasicStatistics "',child.tag,'" has been ignored!')
    assert (len(self.toDo)>0), self.raiseAnError(IOError, 'BasicStatistics needs parameters to work on! Please check input for PP: ' + self.name)

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1: self.raiseAnError(RuntimeError, ' No available Output to collect (run possibly not finished yet)')
    outputDictionary = finishedJob.getEvaluation()[1]
    methodToTest = []
    for key in self.methodsToRun:
      if key not in self.acceptedCalcParam: methodToTest.append(key)
    if isinstance(output,Files.File):
      availExtens = ['xml','csv']
      outputExtension = output.getExt().lower()
      if outputExtension not in availExtens:
        self.raiseAMessage('BasicStatistics did not recognize extension ".'+str(outputExtension)+'" as ".xml", so writing text output...')
      output.setPath(self.__workingDir)
      self.raiseADebug('Writing statistics output in file named ' + output.getAbsFile())
      output.open('w')
      if outputExtension == 'xml':
        self._writeXML(output,outputDictionary,methodToTest)
      else:
        separator = '   ' if outputExtension != 'csv' else ','
        self._writeText(output,outputDictionary,methodToTest,separator)
    elif output.type in ['PointSet','HistorySet']:
      self.raiseADebug('Dumping output in data object named ' + output.name)
      outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
      for ts, outputDict in enumerate(outputResults):
        appendix = '-'+self.pivotParameter+'-'+str(outputDictionary.keys()[ts]) if self.dynamic else ''
        for what in outputDict.keys():
          if what not in self.vectorVals + methodToTest:
            for targetP in outputDict[what].keys():
              self.raiseADebug('Dumping variable ' + targetP + '. Parameter: ' + what + '. Metadata name = ' + targetP + '-' + what)
              output.updateMetadata(targetP + '-' + what + appendix, outputDict[what][targetP])
          else:
            if what not in methodToTest and len(self.allUsedParams) > 1:
              self.raiseADebug('Dumping vector metric',what)
              output.updateMetadata(what.replace("|","-") + appendix, outputDict[what])
        if self.externalFunction:
          self.raiseADebug('Dumping External Function results')
          for what in self.methodsToRun:
            if what not in self.acceptedCalcParam:
              output.updateMetadata(what + appendix, outputDict[what])
              self.raiseADebug('Dumping External Function parameter ' + what)
    else: self.raiseAnError(IOError, 'Output type ' + str(output.type) + ' unknown.')

  def _writeText(self,output,outputDictionary,methodToTest,separator='  '):
    """
      Defines the method for writing the basic statistics to a text file (space and newline delimited)
      @ In, output, File object, file to write to
      @ In, outputDictionary, dict, dictionary of statistics values (or list of the same if self.dynamic)
      @ In, methodToTest, list, strings of methods to test
      @ In, separator, string, optional, separator string (e.g. for csv use ",")
      @ Out, None
    """
    if self.dynamic: output.write('Dynamic BasicStatistics'+ separator+ 'Pivot Parameter' + separator + self.pivotParameter + separator + os.linesep)
    quantitiesToWrite = {}
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    longestParam = max(list(len(param) for param in self.allUsedParams)+[9]) #9 is for 'Metric:'
    # use format functions to make writing matrices easier
    paramFormat = ('{:>'+str(longestParam)+'.'+str(longestParam)+'}').format
    for ts, outputDict in enumerate(outputResults):
      if self.dynamic:
        output.write('Pivot Value' +separator+ str(outputDictionary.keys()[ts]) + os.linesep)
      # do scalars metrics first
      #header
      haveScalars = list(scalar for scalar in self.scalarVals if scalar in outputDict.keys())
      if 'percentile_map' in self.parameters and len(self.parameters['percentile_map']) >0 :
        haveScalars = haveScalars + ['percentile_'+val for val in self.parameters['percentile_map'].values()]
      if len(haveScalars) > 0:
        longestScalar = max(18,max(len(scalar) for scalar in haveScalars))
        valueStrFormat = ('{:^22.22}').format
        valueFormat = '{:+.15e}'.format
        output.write(paramFormat('Metric:') + separator)
        output.write(separator.join(valueStrFormat(scalar) for scalar in haveScalars) + os.linesep)
        #body
        for param in self.allUsedParams:
          output.write(paramFormat(param) + separator)
          values = [None]*len(haveScalars)
          for s,scalar in enumerate(haveScalars):
            if param in outputDict.get(scalar,{}).keys():
              values[s] = valueFormat(outputDict[scalar][param])
            else:
              values[s] = valueStrFormat('---')
          output.write(separator.join(values) + os.linesep)
      # then do vector metrics (matrix style)
      haveVectors = list(vector for vector in self.vectorVals if vector in outputDict.keys())
      for vector in haveVectors:
        #label
        output.write(os.linesep + os.linesep)
        output.write(vector+':'+os.linesep)
        #header
        vecTargets = sorted(outputDict[vector].keys())
        output.write(separator.join(valueStrFormat(v) for v in [' ']+vecTargets)+os.linesep)
        #populate feature list
        vecFeatures = set()
        list(vecFeatures.update(set(outputDict[vector][t].keys())) for t in vecTargets)
        vecFeatures = sorted(list(vecFeatures))
        #body
        for feature in vecFeatures:
          output.write(valueStrFormat(feature)+separator)
          values = [valueStrFormat('---')]*len(vecTargets)
          for t,target in enumerate(vecTargets):
            if feature in outputDict[vector][target].keys():
              values[t] = valueFormat(outputDict[vector][target][feature])
          output.write(separator.join(values)+os.linesep)

  def _writeXML(self,origOutput,outputDictionary,methodToTest):
    """
      Defines the method for writing the basic statistics to a .xml file.
      @ In, origOutput, File object, file to write
      @ In, outputDictionary, dict, dictionary of statistics values
      @ In, methodToTest, list, strings of methods to test
      @ Out, None
    """
    #create XML output with same path as original output
    if origOutput.isOpen(): origOutput.close()
    if self.dynamic:
      output = Files.returnInstance('DynamicXMLOutput',self)
    else:
      output = Files.returnInstance('StaticXMLOutput',self)
    output.initialize(origOutput.getFilename(),self.messageHandler,path=origOutput.getPath())
    output.newTree('BasicStatisticsPP',pivotParam=self.pivotParameter)
    outputResults = [outputDictionary] if not self.dynamic else outputDictionary.values()
    for ts, outputDict in enumerate(outputResults):
      pivotVal = outputDictionary.keys()[ts]
      for t,target in enumerate(self.allUsedParams):
        #do scalars first
        for metric in self.scalarVals:
          #TODO percentile
          if metric == 'percentile':
            for key in outputDict.keys():
              if key.startswith(metric) and target in outputDict[key].keys():
                output.addScalar(target,key,outputDict[key][target],pivotVal=pivotVal)
          elif metric in outputDict.keys() and target in outputDict[metric]:
            output.addScalar(target,metric,outputDict[metric][target],pivotVal=pivotVal)
        #do matrix values
        for metric in self.vectorVals:
          if metric in outputDict.keys() and target in outputDict[metric]:
            output.addVector(target,metric,outputDict[metric][target],pivotVal=pivotVal)

    output.writeFile()

  def __computeVp(self,p,weights):
    """
      Compute the sum of p-th power of weights
      @ In, p, int, the power
      @ In, weights, list or numpy.array, weights
      @ Out, vp, float, the sum of p-th power of weights
    """
    vp = np.sum(np.power(weights,p))
    return vp

  def __computeUnbiasedCorrection(self,order,weightsOrN):
    """
      Compute unbiased correction given weights and momement order
      Reference paper:
      Lorenzo Rimoldini, "Weighted skewness and kurtosis unbiased by sample size", http://arxiv.org/pdf/1304.6564.pdf
      @ In, order, int, moment order
      @ In, weightsOrN, list/numpy.array or int, if list/numpy.array -> weights else -> number of samples
      @ Out, corrFactor, float (order <=3) or tuple of floats (order ==4), the unbiased correction factor
    """
    if order > 4: self.raiseAnError(RuntimeError,"computeUnbiasedCorrection is implemented for order <=4 only!")
    if type(weightsOrN).__name__ not in ['int','int8','int16','int64','int32']:
      if order == 2:
        V1, v1Square, V2 = self.__computeVp(1, weightsOrN), self.__computeVp(1, weightsOrN)**2.0, self.__computeVp(2, weightsOrN)
        corrFactor   = v1Square/(v1Square-V2)
      elif order == 3:
        V1, v1Cubic, V2, V3 = self.__computeVp(1, weightsOrN), self.__computeVp(1, weightsOrN)**3.0, self.__computeVp(2, weightsOrN), self.__computeVp(3, weightsOrN)
        corrFactor   =  v1Cubic/(v1Cubic-3.0*V2*V1+2.0*V3)
      elif order == 4:
        V1, v1Square, V2, V3, V4 = self.__computeVp(1, weightsOrN), self.__computeVp(1, weightsOrN)**2.0, self.__computeVp(2, weightsOrN), self.__computeVp(3, weightsOrN), self.__computeVp(4, weightsOrN)
        numer1 = v1Square*(v1Square**2.0-3.0*v1Square*V2+2.0*V1*V3+3.0*V2**2.0-3.0*V4)
        numer2 = 3.0*v1Square*(2.0*v1Square*V2-2.0*V1*V3-3.0*V2**2.0+3.0*V4)
        denom = (v1Square-V2)*(v1Square**2.0-6.0*v1Square*V2+8.0*V1*V3+3.0*V2**2.0-6.0*V4)
        corrFactor = numer1/denom ,numer2/denom
    else:
      if   order == 2:
        corrFactor   = float(weightsOrN)/(float(weightsOrN)-1.0)
      elif order == 3: corrFactor   = (float(weightsOrN)**2.0)/((float(weightsOrN)-1)*(float(weightsOrN)-2))
      elif order == 4: corrFactor = (float(weightsOrN)*(float(weightsOrN)**2.0-2.0*float(weightsOrN)+3.0))/((float(weightsOrN)-1)*(float(weightsOrN)-2)*(float(weightsOrN)-3)),(3.0*float(weightsOrN)*(2.0*float(weightsOrN)-3.0))/((float(weightsOrN)-1)*(float(weightsOrN)-2)*(float(weightsOrN)-3))
    return corrFactor

  def _computeKurtosis(self,arrayIn,expValue,variance,pbWeight=None):
    """
      Method to compute the Kurtosis (fisher) of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the Kurtosis needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, variance, float, variance of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the Kurtosis of the array of data
    """
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(4,pbWeight) if not self.biased else 1.0
      if not self.biased: result = -3.0 + ((1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,4.0),pbWeight))*unbiasCorr[0]-unbiasCorr[1]*np.power(((1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,2.0),pbWeight))),2.0))/np.power(variance,2.0)
      else              : result = -3.0 + ((1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,4.0),pbWeight))*unbiasCorr)/np.power(variance,2.0)
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(4,len(arrayIn)) if not self.biased else 1.0
      if not self.biased: result = -3.0 + ((1.0/float(len(arrayIn)))*np.sum((arrayIn - expValue)**4)*unbiasCorr[0]-unbiasCorr[1]*(np.average((arrayIn - expValue)**2))**2.0)/(variance)**2.0
      else              : result = -3.0 + ((1.0/float(len(arrayIn)))*np.sum((arrayIn - expValue)**4)*unbiasCorr)/(variance)**2.0
    return result

  def _computeSkewness(self,arrayIn,expValue,variance,pbWeight=None):
    """
      Method to compute the skewness of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the skewness needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, variance, float, variance value of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the skewness of the array of data
    """
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(3,pbWeight) if not self.biased else 1.0
      result = (1.0/self.__computeVp(1,pbWeight))*np.sum(np.dot(np.power(arrayIn - expValue,3.0),pbWeight))*unbiasCorr/np.power(variance,1.5)
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(3,len(arrayIn)) if not self.biased else 1.0
      result = ((1.0/float(len(arrayIn)))*np.sum((arrayIn - expValue)**3)*unbiasCorr)/np.power(variance,1.5)
    return result

  def _computeVariance(self,arrayIn,expValue,pbWeight=None):
    """
      Method to compute the Variance (fisher) of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the Variance needs to be estimated
      @ In, expValue, float, expected value of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, result, float, the Variance of the array of data
    """
    if pbWeight is not None:
      unbiasCorr = self.__computeUnbiasedCorrection(2,pbWeight) if not self.biased else 1.0
      result = (1.0/self.__computeVp(1,pbWeight))*np.average((arrayIn - expValue)**2,weights= pbWeight)*unbiasCorr
    else:
      unbiasCorr = self.__computeUnbiasedCorrection(2,len(arrayIn)) if not self.biased else 1.0
      result = np.average((arrayIn - expValue)**2)*unbiasCorr
    return result

  def _computeSigma(self,arrayIn,variance,pbWeight=None):
    """
      Method to compute the sigma of an array of observations
      @ In, arrayIn, list/numpy.array, the array of values from which the sigma needs to be estimated
      @ In, variance, float, variance of arrayIn
      @ In, pbWeight, list/numpy.array, optional, the reliability weights that correspond to the values in 'array'. If not present, an unweighted approach is used
      @ Out, sigma, float, the sigma of the array of data
    """
    return np.sqrt(variance)

  def _computeWeightedPercentile(self,arrayIn,pbWeight,percent=0.5):
    """
      Method to compute the weighted percentile in a array of data
      @ In, arrayIn, list/numpy.array, the array of values from which the percentile needs to be estimated
      @ In, pbWeight, list/numpy.array, the reliability weights that correspond to the values in 'array'
      @ In, percent, float, the percentile that needs to be computed (between 0.01 and 1.0)
      @ Out, result, float, the percentile
    """
    idxs                   = np.argsort(np.asarray(zip(pbWeight,arrayIn))[:,1])
    # Inserting [0.0,arrayIn[idxs[0]]] is needed when few samples are generated and
    # a percentile that is < that the first pb weight is requested. Otherwise the median
    # is returned (that is wrong).
    sortedWeightsAndPoints = np.insert(np.asarray(zip(pbWeight[idxs],arrayIn[idxs])),0,[0.0,arrayIn[idxs[0]]],axis=0)
    weightsCDF             = np.cumsum(sortedWeightsAndPoints[:,0])
    try:
      index = utils.find_le_index(weightsCDF,percent)
      result = sortedWeightsAndPoints[index,1]
    except ValueError:
      result = np.median(arrayIn)
    return result

  def __runLocal(self, input):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  input, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    pbWeights, pbPresent  = {'realization':None}, False
    if self.externalFunction:
      # there is an external function
      for what in self.methodsToRun:
        outputDict[what] = self.externalFunction.evaluate(what, input['targets'])
        # check if "what" corresponds to an internal method
        if what in self.acceptedCalcParam:
          if what not in ['pearson', 'covariance', 'NormalizedSensitivity', 'VarianceDependentSensitivity', 'sensitivity']:
            if type(outputDict[what]) != dict: self.raiseAnError(IOError, 'BasicStatistics postprocessor: You have overwritten the "' + what + '" method through an external function, it must be a dictionary!!')
          else:
            if type(outputDict[what]) != np.ndarray: self.raiseAnError(IOError, 'BasicStatistics postprocessor: You have overwritten the "' + what + '" method through an external function, it must be a numpy.ndarray!!')
            if len(outputDict[what].shape) != 2    : self.raiseAnError(IOError, 'BasicStatistics postprocessor: You have overwritten the "' + what + '" method through an external function, it must be a 2D numpy.ndarray!!')
    # setting some convenience values
    parameterSet = list(self.allUsedParams)
    if 'metadata' in input.keys(): pbPresent = 'ProbabilityWeight' in input['metadata'].keys() if 'metadata' in input.keys() else False
    if not pbPresent:
      pbWeights['realization'] = None
      if 'metadata' in input.keys():
        if 'SamplerType' in input['metadata'].keys():
          if input['metadata']['SamplerType'][0] != 'MonteCarlo' :
            self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights! Assuming unit weights instead...')
        else:
          self.raiseAWarning('BasicStatistics postprocessor did not detect ProbabilityWeights. Assuming unit weights instead...')
    else:
      pbWeights['realization'] = input['metadata']['ProbabilityWeight']/np.sum(input['metadata']['ProbabilityWeight'])
    #This section should take the probability weight for each sampling variable
    pbWeights['SampledVarsPbWeight'] = {'SampledVarsPbWeight':{}}
    if 'metadata' in input.keys():
      for target in parameterSet:
        if 'ProbabilityWeight-'+target in input['metadata'].keys():
          pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target] = np.asarray(input['metadata']['ProbabilityWeight-'+target])
          pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target][:] = pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target][:]/np.sum(pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][target])

    #establish a dict of indices to parameters and vice versa
    parameter2index = dict((param,p) for p,param in enumerate(input['targets'].keys()))
    for p,param in enumerate(input['targets'].keys()):
      parameter2index[param] = p

    #storage dictionary for skipped metrics
    self.skipped = {}

    #construct a dict of required computations
    needed = dict((metric,set()) for metric in self.scalarVals) #for each metric (keys), the list of parameters we need that value for
    needed.update(dict((metric,{'targets':set(),'features':set()}) for metric in self.vectorVals))
    #percentile is a special exception
    if 'percentile' in needed.keys():
      needed['percentile'] = {}
    #add things requested by the user
    #start by adding the exact request by the user, then add the dependencies
    for metric,params in self.toDo.items():
      #percentile is a special case, and it neither relies on anything nor is relied upon by anything
      if metric == 'percentile':
        for pct,targets in params.items():
          needed[metric][pct] = targets
      elif type(params) == set: #scalar parameter
        needed[metric].update(params)
      elif type(params) == list and type(params[0]) == dict:  # vector parameter
        needed[metric] = {'targets':set(),'features':set()}
        for entry in params:
          needed[metric]['targets'].update(entry['targets'])
          needed[metric]['features'].update(entry['features'])
      else:
        self.raiseAWarning('Unrecognized format for metric "'+metric+'!  Expected "set" or "dict" but got',type(params))
    # variable                     | needs                  | needed for
    # --------------------------------------------------------------------
    # skewness needs               | expectedValue,variance |
    # kurtosis needs               | expectedValue,variance |
    # median needs                 |                        |
    # percentile needs             |                        |
    # maximum needs                |                        |
    # minimum needs                |                        |
    # covariance needs             |                        | pearson,VarianceDependentSensitivity,NormalizedSensitivity
    # NormalizedSensitivity        | covariance,VarDepSens  |
    # VarianceDependentSensitivity | covariance             | NormalizedSensitivity
    # sensitivity needs            |                        |
    # pearson needs                | covariance             |
    # sigma needs                  | variance               | variationCoefficient
    # variance                     | expectedValue          | sigma, skewness, kurtosis
    # expectedValue                |                        | variance, variationCoefficient, skewness, kurtosis
    needed['sigma'].update(needed.get('variationCoefficient'))
    needed['variance'].update(needed.get('sigma',set()))
    needed['expectedValue'].update(needed.get('sigma',set()))
    needed['expectedValue'].update(needed.get('variationCoefficient',set()))
    needed['expectedValue'].update(needed.get('variance',set()))
    needed['expectedValue'].update(needed.get('skewness',set()))
    needed['expectedValue'].update(needed.get('kurtosis',set()))
    if 'NormalizedSensitivity' in needed.keys():
      needed['expectedValue'].update(needed['NormalizedSensitivity']['targets'])
      needed['expectedValue'].update(needed['NormalizedSensitivity']['features'])
      needed['covariance']['targets'].update(needed['NormalizedSensitivity']['targets'])
      needed['covariance']['features'].update(needed['NormalizedSensitivity']['features'])
      needed['VarianceDependentSensitivity']['targets'].update(needed['NormalizedSensitivity']['targets'])
      needed['VarianceDependentSensitivity']['features'].update(needed['NormalizedSensitivity']['features'])
    if 'pearson' in needed.keys():
      needed['covariance']['targets'].update(needed['pearson']['targets'])
      needed['covariance']['features'].update(needed['pearson']['features'])
    if 'VarianceDependentSensitivity' in needed.keys():
      needed['covariance']['targets'].update(needed['VarianceDependentSensitivity']['targets'])
      needed['covariance']['features'].update(needed['VarianceDependentSensitivity']['features'])
    #
    # BEGIN actual calculations
    #
    calculations = {}
    # do things in order to preserve prereqs
    # TODO many of these could be sped up through vectorization
    # TODO additionally, this could be done with less code duplication, probably
    #################
    # SCALAR VALUES #
    #################
    def startMetric(metric):
      """
        Common starting for each metric calculation.
        @ In, metric, string, name of metric
        @ Out, None
      """
      if len(needed[metric])>0:
        self.raiseADebug('Starting "'+metric+'"...')
        calculations[metric]={}
    #
    # samples
    #
    metric = 'samples'
    startMetric(metric)
    for targetP in needed[metric]:
      calculations[metric][targetP] = len(input['targets'].values()[0])
    #
    # expected value
    #
    metric = 'expectedValue'
    startMetric(metric)
    for targetP in needed[metric]:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
        calculations[metric][targetP] = np.average(input['targets'][targetP], weights = relWeight)
      else:
        relWeight  = None
        calculations[metric][targetP] = np.mean(input['targets'][targetP])
    #
    # variance
    #
    metric = 'variance'
    startMetric(metric)
    for targetP in needed[metric]:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
      else:
        relWeight  = None
      calculations[metric][targetP] = self._computeVariance(input['targets'][targetP],calculations['expectedValue'][targetP],pbWeight=relWeight)
      #sanity check
      if (calculations[metric][targetP] == 0):
        self.raiseAWarning('The variable: ' + targetP + ' has zero variance! Please check your input in PP: ' + self.name)
    #
    # sigma
    #
    metric = 'sigma'
    startMetric(metric)
    for targetP in needed[metric]:
      if calculations['variance'][targetP] == 0:#np.Infinity:
        self.raiseAWarning('The variable: ' + targetP + ' has zero sigma! Please check your input in PP: ' + self.name)
        calculations[metric][targetP] = 0.0
      else:
        calculations[metric][targetP] = self._computeSigma(input['targets'][targetP],calculations['variance'][targetP])
    #
    # coeff of variation (sigma/mu)
    #
    metric = 'variationCoefficient'
    startMetric(metric)
    for targetP in needed[metric]:
      if calculations['expectedValue'][targetP] == 0:
        self.raiseAWarning('Expected Value for ' + targetP + ' is zero! Variation Coefficient cannot be calculated, so setting as infinite.')
        calculations[metric][targetP] = np.Infinity
      else:
        calculations[metric][targetP] = calculations['sigma'][targetP]/calculations['expectedValue'][targetP]
    #
    # skewness
    #
    metric = 'skewness'
    startMetric(metric)
    for targetP in needed[metric]:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
      else:
        relWeight  = None
      calculations[metric][targetP] = self._computeSkewness(input['targets'][targetP],calculations['expectedValue'][targetP],calculations['variance'][targetP],pbWeight=relWeight)
    #
    # kurtosis
    #
    metric = 'kurtosis'
    startMetric(metric)
    for targetP in needed[metric]:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
      else:
        relWeight  = None
      calculations[metric][targetP] = self._computeKurtosis(input['targets'][targetP],calculations['expectedValue'][targetP],calculations['variance'][targetP],pbWeight=relWeight)
    #
    # median
    #
    metric = 'median'
    startMetric(metric)
    for targetP in needed[metric]:
      if pbPresent:
        relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
        calculations[metric][targetP] = self._computeWeightedPercentile(input['targets'][targetP],relWeight,percent=0.5)
      else:
        calculations[metric][targetP] = np.median(input['targets'][targetP])
    #
    # maximum
    #
    metric = 'maximum'
    startMetric(metric)
    for targetP in needed[metric]:
      calculations[metric][targetP] = np.amax(input['targets'][targetP])
    #
    # minimum
    #
    metric = 'minimum'
    startMetric(metric)
    for targetP in needed[metric]:
      calculations[metric][targetP] = np.amin(input['targets'][targetP])
    #
    # percentile
    #
    metric = 'percentile'
    self.raiseADebug('Starting "'+metric+'"...')
    for percent,targets in needed[metric].items():
      self.raiseADebug('...',str(percent),'...')
      label = metric+'_'+self.parameters['percentile_map'][percent]
      calculations[label] = {}
      for targetP in targets:
        if pbPresent: relWeight  = pbWeights['realization'] if targetP not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][targetP]
        calculations[label][targetP] = np.percentile(input['targets'][targetP], percent) if not pbPresent else self._computeWeightedPercentile(input['targets'][targetP],relWeight,percent=float(percent)/100.0)
    #################
    # VECTOR VALUES #
    #################
    #
    # sensitivity matrix
    #
    def startVector(metric):
      """
        Common method among all metrics for establishing parameters
        @ In, metric, string, the name of the statistics metric to calculate
        @ Out, targets, list(str), list of target parameter names (evaluate metrics for these)
        @ Out, features, list(str), list of feature parameter names (evaluate with respect to these)
        @ Out, skip, bool, if True it means either features or parameters were missing, so don't calculate anything
      """
      # default to skipping, change that if we find criteria
      targets = []
      features = []
      skip = True
      allParams = set(needed[metric]['targets'])
      allParams.update(set(needed[metric]['features']))
      if len(needed[metric]['targets'])>0 and len(allParams)>=2:
        self.raiseADebug('Starting "'+metric+'"...')
        calculations[metric]={}
        targets = needed[metric]['targets']
        features = needed[metric]['features']
        skip = False #True only if we don't have targets and features
        if len(features)<1:
          self.raiseAWarning('No features specified for <'+metric+'>!  Please specify features in a <features> node (see the manual).  Skipping...')
          skip = True
      elif len(needed[metric]['targets']) == 0:
        #unrequested, no message needed
        pass
      elif len(allParams) < 2:
        #insufficient target/feature combinations (usually when only 1 target and 1 feature, and they are the same)
        self.raiseAWarning('A total of',len(allParams),'were provided for metric',metric,'but at least 2 are required!  Skipping...')
      if skip:
        if metric not in self.skipped.keys():
          self.skipped[metric] = {}
        self.skipped[metric].update(needed[metric])
      return targets,features,skip

    metric = 'sensitivity'
    targets,features,skip = startVector(metric)
    #NOTE sklearn expects the transpose of what we usually do in RAVEN, so #samples by #features
    if not skip:
      #for sensitivity matrix, we don't use numpy/scipy methods to calculate matrix operations,
      #so we loop over targets and features
      for t,target in enumerate(targets):
        calculations[metric][target] = {}
        targetVals = input['targets'][target]
        #don't do self-sensitivity
        inpSamples = np.atleast_2d(np.asarray(list(input['targets'][f] for f in features if f!=target))).T
        useFeatures = list(f for f in features if f != target)
        #use regressor coefficients as sensitivity
        regressDict = dict(zip(useFeatures, LinearRegression().fit(inpSamples,targetVals).coef_))
        for f,feature in enumerate(features):
          calculations[metric][target][feature] = 1.0 if feature==target else regressDict[feature]
    #
    # covariance matrix
    #
    metric = 'covariance'
    targets,features,skip = startVector(metric)
    if not skip:
      # because the C implementation is much faster than picking out individual values,
      #   we do the full covariance matrix with all the targets and features.
      # FIXME adding an alternative for users to choose pick OR do all, defaulting to something smart
      #   dependent on the percentage of the full matrix desired, would be better.
      # IF this is fixed, make sure all the features and targets are requested for all the metrics
      #   dependent on this metric
      params = list(set(targets).union(set(features)))
      paramSamples = np.zeros((len(params), utils.first(input['targets'].values()).size))
      pbWeightsList = [None]*len(input['targets'].keys())
      for p,param in enumerate(params):
        dataIndex = parameter2index[param]
        paramSamples[p,:] = input['targets'][param][:]
        pbWeightsList[p] = pbWeights['realization'] if param not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][param]
      pbWeightsList.append(pbWeights['realization'])
      if None in pbWeightsList:
        covar = self.covariance(paramSamples)
      else:
        covar = self.covariance(paramSamples, weights = pbWeightsList)
      calculations[metric]['matrix'] = covar
      calculations[metric]['params'] = params

    def getCovarianceSubset(desired):
      """
        @ In, desired, list(str), list of parameters to extract from covariance matrix
        @ Out, reducedSecond, np.array, reduced covariance matrix
        @ Out, wantedParams, list(str), parameter labels for reduced covar matrix
      """
      wantedIndices = list(calculations['covariance']['params'].index(d) for d in desired)
      wantedParams = list(calculations['covariance']['params'][i] for i in wantedIndices)
      #retain rows, colums
      reducedFirst = calculations['covariance']['matrix'][wantedIndices]
      reducedSecond = reducedFirst[:,wantedIndices]
      return reducedSecond, wantedParams
    #
    # pearson matrix
    #
    # see comments in covariance for notes on C implementation
    metric = 'pearson'
    targets,features,skip = startVector(metric)
    if not skip:
      params = list(set(targets).union(set(features)))
      reducedCovar,reducedParams = getCovarianceSubset(params)
      calculations[metric]['matrix'] = self.corrCoeff(reducedCovar)
      calculations[metric]['params'] = reducedParams
    #
    # VarianceDependentSensitivity matrix
    # The formula for this calculation is coming from: http://www.math.uah.edu/stat/expect/Matrices.html
    # The best linear predictor: L(Y|X) = expectedValue(Y) + cov(Y,X) * [vc(X)]^(-1) * [X-expectedValue(X)]
    # where Y is a vector of outputs, and X is a vector of inputs, cov(Y,X) is the covariance matrix of Y and X,
    # vc(X) is the covariance matrix of X with itself.
    # The variance dependent sensitivity matrix is defined as: cov(Y,X) * [vc(X)]^(-1)
    #
    metric = 'VarianceDependentSensitivity'
    targets,features,skip = startVector(metric)
    if not skip:
      params = list(set(targets).union(set(features)))
      reducedCovar,reducedParams = getCovarianceSubset(params)
      inputSamples = np.zeros((len(params),utils.first(input['targets'].values()).size))
      pbWeightsList = [None]*len(params)
      for p,param in enumerate(reducedParams):
        inputSamples[p,:] = input['targets'][param][:]
        pbWeightsList[p] = pbWeights['realization'] if param not in pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'].keys() else pbWeights['SampledVarsPbWeight']['SampledVarsPbWeight'][param]
      pbWeightsList.append(pbWeights['realization'])
      for p,param in enumerate(reducedParams):
        calculations[metric][param] = {}
        targCoefs = list(r for r in reducedParams if r!=param)
        inpParams = np.delete(inputSamples,p,axis=0)
        inpCovMatrix = np.delete(reducedCovar,p,axis=0)
        inpCovMatrix = np.delete(inpCovMatrix,p,axis=1)
        outInpCov = np.delete(reducedCovar[p,:],p)
        sensCoefDict = dict(zip(targCoefs,np.dot(outInpCov,np.linalg.pinv(inpCovMatrix))))
        for f,feature in enumerate(reducedParams):
          if param == feature:
            calculations[metric][param][feature] = 1.0
          else:
            calculations[metric][param][feature] = sensCoefDict[feature]
    #
    # Normalized variance dependent sensitivity matrix
    # variance dependent sensitivity  normalized by the mean (% change of output)/(% change of input)
    #
    metric = 'NormalizedSensitivity'
    targets,features,skip = startVector(metric)
    if not skip:
      reducedCovar,reducedParams = getCovarianceSubset(params)
      for p,param in enumerate(reducedParams):
        calculations[metric][param] = {}
        for f,feature in enumerate(reducedParams):
          expValueRatio = calculations['expectedValue'][feature]/calculations['expectedValue'][param]
          calculations[metric][param][feature] = calculations['VarianceDependentSensitivity'][param][feature]*expValueRatio

    #collect only the requested calculations
    outputDict = {}
    for metric,params in self.toDo.items():
      #TODO someday we might need to expand the "skipped" check to include scalars, but for now
      #   the only reason to skip is if an invalid matrix is requested
      #if percentile, special treatment
      if metric == 'percentile':
        for pct,targets in params.items():
          label = 'percentile_'+self.parameters['percentile_map'][pct]
          outputDict[label] = dict((target,calculations[label][target]) for target in targets)
      #if other scalar, just report the result
      elif metric in self.scalarVals:
        outputDict[metric] = dict((target,calculations[metric][target]) for target in params)
      #if a matrix block, extract desired values
      else:
        if metric in ['pearson','covariance']:
          outputDict[metric] = {}
          for entry in params:
            #check if it was skipped for some reason
            if entry == self.skipped.get(metric,None):
              self.raiseADebug('Metric',metric,'was skipped for parameters',entry,'!  See warnings for details.  Ignoring...')
              continue
            for target in entry['targets']:
              if target not in outputDict[metric].keys():
                outputDict[metric][target] = {}
              targetIndex = calculations[metric]['params'].index(target)
              for feature in entry['features']:
                featureIndex = calculations[metric]['params'].index(feature)
                outputDict[metric][target][feature] = calculations[metric]['matrix'][targetIndex,featureIndex]
        #if matrix but stored in dictionaries, just grab the values
        elif metric in ['sensitivity','NormalizedSensitivity','VarianceDependentSensitivity']:
          outputDict[metric] = {}
          for entry in params:
            #check if it was skipped for some reason
            if entry == self.skipped.get(metric,None):
              self.raiseADebug('Metric',metric,'was skipped for parameters',entry,'!  See warnings for details.  Ignoring...')
              continue
            for target in entry['targets']:
              outputDict[metric][target] = dict((feature,calculations[metric][target][feature]) for feature in entry['features'])

    # print on screen
    methodToTest = []
    for key in self.methodsToRun:
      if key not in self.acceptedCalcParam: methodToTest.append(key)
    self.printToScreen(outputDict)
    return outputDict

  def printToScreen(self,outputDict):
    """
      Prints all results of BasicStatistics to screen.
      @ In, outputDict, dict, dictionary of results
      @ Out, None
    """
    self.raiseADebug('BasicStatistics ' + str(self.name) + 'results:')
    for metric,valueDict in outputDict.items():
      self.raiseADebug('BasicStatistics Metric:',metric)
      if metric in self.scalarVals or metric.startswith('percentile'):
        for target,value in valueDict.items():
          self.raiseADebug('   ',target+':',value)
      elif metric in self.vectorVals:
        for target,wrt in valueDict.items():
          self.raiseADebug('   ',target,'with respect to:')
          for feature,value in wrt.items():
            self.raiseADebug('     ',feature+':',value)
      else:
        self.raiseADebug('   ',valueDict)

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case, it computes all the requested statistical FOMs
      @ In,  inputIn, object, object contained the data to process. (inputToInternal output)
      @ Out, outputDict, dict, Dictionary containing the results
    """
    input = self.inputToInternal(inputIn)
    if not self.dynamic: outputDict = self.__runLocal(input)
    else:
      # time dependent (actually pivot-dependent)
      outputDict = OrderedDict()
      self.raiseADebug('BasicStatistics Pivot-Dependent output:')
      for pivotParamValue in input['timeDepData'].keys():
        self.raiseADebug('Pivot Parameter Value: ' + str(pivotParamValue))
        outputDict[pivotParamValue] = self.__runLocal(input['timeDepData'][pivotParamValue])
    return outputDict

  def covariance(self, feature, weights = None, rowVar = 1):
    """
      This method calculates the covariance Matrix for the given data.
      Unbiased unweighted covariance matrix, weights is None, bias is 0 (default)
      Biased unweighted covariance matrix,   weights is None, bias is 1
      Unbiased weighted covariance matrix,   weights is not None, bias is 0
      Biased weighted covariance matrix,     weights is not None, bias is 1
      can be calculated depending on the selection of the inputs.
      @ In,  feature, list/numpy.array, [#targets,#samples]  features' samples
      @ In,  weights, list of list/numpy.array, optional, [#targets,#samples,realizationWeights]  reliability weights, and the last one in the list is the realization weights. Default is None
      @ In,  rowVar, int, optional, If rowVar is non-zero, then each row represents a variable,
                                    with samples in the columns. Otherwise, the relationship is transposed. Default=1
      @ Out, covMatrix, list/numpy.array, [#targets,#targets] the covariance matrix
    """
    X = np.array(feature, ndmin = 2, dtype = np.result_type(feature, np.float64))
    w = np.zeros(feature.shape, dtype = np.result_type(feature, np.float64))
    if X.shape[0] == 1:
      rowVar = 1
    if rowVar:
      N = X.shape[1]
      featuresNumber = X.shape[0]
      axis = 0
      for myIndex in range(featuresNumber):
        if weights is None:
          w[myIndex,:] = np.ones(N)/float(N)
        else:
          w[myIndex,:] = np.array(weights[myIndex],dtype=np.result_type(feature, np.float64))[:] if weights is not None else np.ones(len(w[myIndex,:]),dtype =np.result_type(feature, np.float64))[:]
    else:
      N = X.shape[0]
      featuresNumber = X.shape[1]
      axis = 1
      for myIndex in range(featuresNumber):
        if weights is None:
          w[myIndex,:] = np.ones(N)/float(N)
        else:
          w[:,myIndex] = np.array(weights[myIndex], dtype=np.result_type(feature, np.float64))[:] if weights is not None else np.ones(len(w[:,myIndex]),dtype=np.result_type(feature, np.float64))[:]
    realizationWeights = weights[-1] if weights is not None else np.ones(N)/float(N)
    if N <= 1:
      self.raiseAWarning("Degrees of freedom <= 0")
      return np.zeros((featuresNumber,featuresNumber), dtype = np.result_type(feature, np.float64))
    diff = X - np.atleast_2d(np.average(X, axis = 1 - axis, weights = w)).T
    covMatrix = np.ones((featuresNumber,featuresNumber), dtype = np.result_type(feature, np.float64))
    for myIndex in range(featuresNumber):
      for myIndexTwo in range(featuresNumber):
        # The weights that are used here should represent the joint probability (P(x,y)).
        # Since I have no way yet to compute the joint probability with weights only (eventually I can think to use an estimation of the P(x,y) computed through a 2D histogram construction and weighted a posteriori with the 1-D weights),
        # I decided to construct a weighting function that is defined as Wi = (2.0*Wi,x*Wi,y)/(Wi,x+Wi,y) that respects the constrains of the
        # covariance (symmetric and that the diagonal is == variance) but that is completely arbitrary and for that not used. As already mentioned, I need the joint probability to compute the E[XY] = integral[xy*p(x,y)dxdy]. Andrea
        # for now I just use the realization weights
        #jointWeights = (2.0*weights[myIndex][:]*weights[myIndexTwo][:])/(weights[myIndex][:]+weights[myIndexTwo][:])
        #jointWeights = jointWeights[:]/np.sum(jointWeights)
        if myIndex == myIndexTwo:
          jointWeights = w[myIndex]/np.sum(w[myIndex])
        else:
          jointWeights = realizationWeights/np.sum(realizationWeights)
        fact = self.__computeUnbiasedCorrection(2,jointWeights) if not self.biased else 1.0/np.sum(jointWeights)
        covMatrix[myIndex,myIndexTwo] = np.sum(diff[:,myIndex]*diff[:,myIndexTwo]*jointWeights[:]*fact) if not rowVar else np.sum(diff[myIndex,:]*diff[myIndexTwo,:]*jointWeights[:]*fact)
    return covMatrix

  def corrCoeff(self, covM):
    """
      This method calculates the correlation coefficient Matrix (pearson) for the given data.
      Unbiased unweighted covariance matrix, weights is None, bias is 0 (default)
      Biased unweighted covariance matrix,   weights is None, bias is 1
      Unbiased weighted covariance matrix,   weights is not None, bias is 0
      Biased weighted covariance matrix,     weights is not None, bias is 1
      can be calcuated depending on the selection of the inputs.
      @ In,  covM, list/numpy.array, [#targets,#targets] covariance matrix
      @ Out, corrMatrix, list/numpy.array, [#targets,#targets] the correlation matrix
    """
    try:
      d = np.diag(covM)
      corrMatrix = covM / np.sqrt(np.multiply.outer(d, d))
    except ValueError:  # scalar covariance
      # nan if incorrect value (nan, inf, 0), 1 otherwise
      corrMatrix = covM / covM
    # to prevent numerical instability
    return corrMatrix

#
#

class LimitSurfaceInput(InputData.ParameterInput):
  """
    Class for reading limit surface block
  """

LimitSurfaceInput.createClass("PostProcessor", False, baseNode=ModelInput)
ParametersInput = InputData.parameterInputFactory("parameters", contentType=InputData.StringType)
LimitSurfaceInput.addSub(ParametersInput)
ToleranceInput = InputData.parameterInputFactory("tolerance", contentType=InputData.FloatType)
LimitSurfaceInput.addSub(ToleranceInput)
SideInput = InputData.parameterInputFactory("side", contentType=InputData.StringType)
LimitSurfaceInput.addSub(SideInput)

#
class LimitSurface(BasePostProcessor):
  """
    LimitSurface filter class. It computes the limit surface associated to a dataset
  """

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self,messageHandler)
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
    self.gridEntity        = None
    self.bounds            = None
    self.jobHandler        = None
    self.transfMethods     = {}
    self.addAssemblerObject('ROM','-1', True)
    self.addAssemblerObject('Function','1')
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

  def inputToInternal(self, currentInp):
    """
      Method to convert an input object into the internal format that is
      understandable by this pp.
      @ In, currentInput, object, an object that needs to be converted
      @ Out, inputDict, dict, the resulting dictionary containing features and response
    """
    # each post processor knows how to handle the coming inputs. The BasicStatistics postprocessor accept all the input type (files (csv only), hdf5 and dataobjects
    if type(currentInp) == list: currentInput = currentInp[-1]
    else                       : currentInput = currentInp
    if type(currentInp) == dict:
      if 'targets' in currentInput.keys(): return
    inputDict = {'targets':{}, 'metadata':{}}
    #FIXME I don't think this is checking for files, HDF5 and dataobjects
    if hasattr(currentInput,'type'):
      inType = currentInput.type
    else:
      self.raiseAnError(IOError, self, 'LimitSurface postprocessor accepts files,HDF5,Data(s) only! Got ' + str(type(currentInput)))
    if isinstance(currentInp,Files.File):
      if currentInput.subtype == 'csv': pass
      #FIXME else?  This seems like hollow code right now.
    if inType == 'HDF5': pass  # to be implemented
    if inType in ['PointSet']:
      for targetP in self.parameters['targets']:
        if   targetP in currentInput.getParaKeys('input'): inputDict['targets'][targetP] = currentInput.getParam('input' , targetP)
        elif targetP in currentInput.getParaKeys('output'): inputDict['targets'][targetP] = currentInput.getParam('output', targetP)
      inputDict['metadata'] = currentInput.getAllMetadata()
    # to be added
    return inputDict

  def _initializeLSpp(self, runInfo, inputs, initDict):
    """
      Method to initialize the LS post processor (create grid, etc.)
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.gridEntity = GridEntities.returnInstance("MultiGridEntity",self,self.messageHandler)
    self.__workingDir     = runInfo['WorkingDir']
    self.externalFunction = self.assemblerDict['Function'][0][3]
    if 'ROM' not in self.assemblerDict.keys():
      self.ROM = LearningGate.returnInstance('SupervisedGate','SciKitLearn', self, **{'SKLtype':'neighbors|KNeighborsClassifier',"n_neighbors":1, 'Features':','.join(list(self.parameters['targets'])), 'Target':[self.externalFunction.name]})
    else: self.ROM = self.assemblerDict['ROM'][0][3]
    self.ROM.reset()
    self.indexes = -1
    for index, inp in enumerate(self.inputs):
      if type(inp).__name__ in ['str', 'bytes', 'unicode']: self.raiseAnError(IOError, 'LimitSurface PostProcessor only accepts Data(s) as inputs!')
      if inp.type == 'PointSet': self.indexes = index
    if self.indexes == -1: self.raiseAnError(IOError, 'LimitSurface PostProcessor needs a PointSet as INPUT!!!!!!')
    else:
      # check if parameters are contained in the data
      inpKeys = self.inputs[self.indexes].getParaKeys("inputs")
      outKeys = self.inputs[self.indexes].getParaKeys("outputs")
      self.paramType = {}
      for param in self.parameters['targets']:
        if param not in inpKeys + outKeys: self.raiseAnError(IOError, 'LimitSurface PostProcessor: The param ' + param + ' not contained in Data ' + self.inputs[self.indexes].name + ' !')
        if param in inpKeys: self.paramType[param] = 'inputs'
        else:                self.paramType[param] = 'outputs'
    if self.bounds == None:
      self.bounds = {"lowerBounds":{},"upperBounds":{}}
      for key in self.parameters['targets']:
        self.bounds["lowerBounds"][key], self.bounds["upperBounds"][key] = min(self.inputs[self.indexes].getParam(self.paramType[key],key,nodeId = 'RecontructEnding')), max(self.inputs[self.indexes].getParam(self.paramType[key],key,nodeId = 'RecontructEnding'))
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
      self.functionValue.update(inp['inputs' ])
      self.functionValue.update(inp['outputs'])
    else:
      self.functionValue.update(inp.getParametersValues('inputs', nodeId = 'RecontructEnding'))
      self.functionValue.update(inp.getParametersValues('outputs', nodeId = 'RecontructEnding'))
    # recovery the index of the last function evaluation performed
    if self.externalFunction.name in self.functionValue.keys(): indexLast = len(self.functionValue[self.externalFunction.name]) - 1
    else                                                      : indexLast = -1
    # index of last set of point tested and ready to perform the function evaluation
    indexEnd = len(self.functionValue[self.axisName[0]]) - 1
    tempDict = {}
    if self.externalFunction.name in self.functionValue.keys():
      self.functionValue[self.externalFunction.name] = np.append(self.functionValue[self.externalFunction.name], np.zeros(indexEnd - indexLast))
    else: self.functionValue[self.externalFunction.name] = np.zeros(indexEnd + 1)

    for myIndex in range(indexLast + 1, indexEnd + 1):
      for key, value in self.functionValue.items(): tempDict[key] = value[myIndex]
      self.functionValue[self.externalFunction.name][myIndex] = self.externalFunction.evaluate('residuumSign', tempDict)
      if abs(self.functionValue[self.externalFunction.name][myIndex]) != 1.0: self.raiseAnError(IOError, 'LimitSurface: the function evaluation of the residuumSign method needs to return a 1 or -1!')
      if type(inp) != dict:
        if self.externalFunction.name in inp.getParaKeys('inputs'): inp.self.updateInputValue (self.externalFunction.name, self.functionValue[self.externalFunction.name][myIndex])
        if self.externalFunction.name in inp.getParaKeys('output'): inp.self.updateOutputValue(self.externalFunction.name, self.functionValue[self.externalFunction.name][myIndex])
      else:
        if self.externalFunction.name in inp['inputs' ].keys(): inp['inputs' ][self.externalFunction.name] = np.concatenate((inp['inputs'][self.externalFunction.name],np.asarray(self.functionValue[self.externalFunction.name][myIndex])))
        if self.externalFunction.name in inp['outputs'].keys(): inp['outputs'][self.externalFunction.name] = np.concatenate((inp['outputs'][self.externalFunction.name],np.asarray(self.functionValue[self.externalFunction.name][myIndex])))
    if np.sum(self.functionValue[self.externalFunction.name]) == float(len(self.functionValue[self.externalFunction.name])) or np.sum(self.functionValue[self.externalFunction.name]) == -float(len(self.functionValue[self.externalFunction.name])):
      if raiseErrorIfNotFound: self.raiseAnError(ValueError, 'LimitSurface: all the Function evaluations brought to the same result (No Limit Surface has been crossed...). Increase or change the data set!')
      else                   : self.raiseAWarning('LimitSurface: all the Function evaluations brought to the same result (No Limit Surface has been crossed...)!')
    #printing----------------------
    self.raiseADebug('LimitSurface: Mapping of the goal function evaluation performed')
    self.raiseADebug('LimitSurface: Already evaluated points and function values:')
    keyList = list(self.functionValue.keys())
    self.raiseADebug(','.join(keyList))
    for index in range(indexEnd + 1):
      self.raiseADebug(','.join([str(self.functionValue[key][index]) for key in keyList]))
    #printing----------------------
    tempDict = {}
    for name in self.axisName: tempDict[name] = np.asarray(self.functionValue[name])
    tempDict[self.externalFunction.name] = self.functionValue[self.externalFunction.name]
    self.ROM.train(tempDict)
    self.raiseADebug('LimitSurface: Training performed')
    self.raiseADebug('LimitSurface: Training finished')

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the LS pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
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
    if "parameters" not in dictIn.keys()             : self.raiseAnError(IOError, 'No Parameters specified in "dictIn" dictionary !!!!')
    if "name"                  in dictIn.keys()      : self.name          = dictIn["name"]
    if type(dictIn["parameters"]).__name__ == "list" : self.parameters['targets'] = dictIn["parameters"]
    else                                             : self.parameters['targets'] = dictIn["parameters"].split(",")
    if "bounds"                in dictIn.keys()      : self.bounds        = dictIn["bounds"]
    if "transformationMethods" in dictIn.keys()      : self.transfMethods = dictIn["transformationMethods"]
    if "verbosity"             in dictIn.keys()      : self.verbosity     = dictIn['verbosity']
    if "side"                  in dictIn.keys()      : self.lsSide        = dictIn["side"]
    if "tolerance"             in dictIn.keys()      : self.tolerance     = float(dictIn["tolerance"])
    if self.lsSide not in ["negative", "positive", "both"]: self.raiseAnError(IOError, 'Computation side can be positive, negative, both only !!!!')

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
    if nodeName == None  : testMatrix = self.testMatrix[self.name]
    elif nodeName =="all":
      if exceptionGrid == None: testMatrix = self.testMatrix
      else:
        returnDict = OrderedDict()
        wantedKeys = list(self.testMatrix.keys())
        wantedKeys.pop(wantedKeys.index(exceptionGrid))
        for key in wantedKeys: returnDict[key] = self.testMatrix[key]
        testMatrix = returnDict
    else                 : testMatrix = self.testMatrix[nodeName]
    return testMatrix

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = LimitSurfaceInput()
    paramInput.parseNode(xmlNode)
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
    if finishedJob.getEvaluation() == -1: self.raiseAnError(RuntimeError, 'No available Output to collect (Run probably is not finished yet)')
    self.raiseADebug(str(finishedJob.getEvaluation()))
    limitSurf = finishedJob.getEvaluation()[1]
    if limitSurf[0] is not None:
      for varName in output.getParaKeys('inputs'):
        for varIndex in range(len(self.axisName)):
          if varName == self.axisName[varIndex]:
            output.removeInputValue(varName)
            for value in limitSurf[0][:, varIndex]: output.updateInputValue(varName, copy.copy(value))
      output.removeOutputValue(self.externalFunction.name)
      for value in limitSurf[1]: output.updateOutputValue(self.externalFunction.name, copy.copy(value))

  def refineGrid(self,refinementSteps=2):
    """
      Method to refine the internal grid based on the limit surface previously computed
      @ In, refinementSteps, int, optional, number of refinement steps
      @ Out, None
    """
    cellIds = self.gridEntity.retrieveCellIds([self.listSurfPointNegative,self.listSurfPointPositive],self.name)
    if self.getLocalVerbosity() == 'debug': self.raiseADebug("Limit Surface cell IDs are: \n"+ " \n".join([str(cellID) for cellID in cellIds]))
    self.raiseAMessage("Number of cells to be refined are "+str(len(cellIds))+". RefinementSteps = "+str(max([refinementSteps,2]))+"!")
    self.gridEntity.refineGrid({"cellIDs":cellIds,"refiningNumSteps":int(max([refinementSteps,2]))})
    for nodeName in self.gridEntity.getAllNodesNames(self.name):
      if nodeName != self.name: self.testMatrix[nodeName] = np.zeros(self.gridEntity.returnParameter("gridShape",nodeName))

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
      try   : allGridNames.pop(allGridNames.index(exceptionGrid))
      except: pass
    self.surfPoint, evaluations, listSurfPoint = OrderedDict().fromkeys(allGridNames), OrderedDict().fromkeys(allGridNames) ,OrderedDict().fromkeys(allGridNames)
    for nodeName in allGridNames:
      #if skipMainGrid == True and nodeName == self.name: continue
      self.testMatrix[nodeName] = np.zeros(self.gridEntity.returnParameter("gridShape",nodeName))
      self.gridCoord[nodeName] = self.gridEntity.returnGridAsArrayOfCoordinates(nodeName=nodeName)
      tempDict ={}
      for  varId, varName in enumerate(self.axisName): tempDict[varName] = self.gridCoord[nodeName][:,varId]
      self.testMatrix[nodeName].shape     = (self.gridCoord[nodeName].shape[0])                       #rearrange the grid matrix such as is an array of values
      self.testMatrix[nodeName][:]        = self.ROM.evaluate(tempDict)[self.externalFunction.name]   #get the prediction on the testing grid
      self.testMatrix[nodeName].shape     = self.gridEntity.returnParameter("gridShape",nodeName)     #bring back the grid structure
      self.gridCoord[nodeName].shape      = self.gridEntity.returnParameter("gridCoorShape",nodeName) #bring back the grid structure
      self.raiseADebug('LimitSurface: Prediction performed')
      # here next the points that are close to any change are detected by a gradient (it is a pre-screener)
      toBeTested = np.squeeze(np.dstack(np.nonzero(np.sum(np.abs(np.gradient(self.testMatrix[nodeName])), axis = 0))))
      #printing----------------------
      self.raiseADebug('LimitSurface:  Limit surface candidate points')
      if self.getLocalVerbosity() == 'debug':
        for coordinate in np.rollaxis(toBeTested, 0):
          myStr = ''
          for iVar, varnName in enumerate(self.axisName): myStr += varnName + ': ' + str(coordinate[iVar]) + '      '
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
      if self.getLocalVerbosity() == 'debug':
        if len(listSurfPoint[nodeName]) > 0: self.raiseADebug('LimitSurface: Limit surface points:')
        for coordinate in listSurfPoint[nodeName]:
          myStr = ''
          for iVar, varnName in enumerate(self.axisName): myStr += varnName + ': ' + str(coordinate[iVar]) + '      '
          self.raiseADebug('LimitSurface: ' + myStr + '  value: ' + str(self.testMatrix[nodeName][tuple(coordinate)]))
      # if the number of point on the limit surface is > than zero than save it
      if len(listSurfPoint[nodeName]) > 0:
        self.surfPoint[nodeName] = np.ndarray((len(listSurfPoint[nodeName]), self.nVar))
        evaluations[nodeName] = np.concatenate((-np.ones(nNegPoints), np.ones(nPosPoints)), axis = 0)
        for pointID, coordinate in enumerate(listSurfPoint[nodeName]):
          self.surfPoint[nodeName][pointID, :] = self.gridCoord[nodeName][tuple(coordinate)]
    if self.name != exceptionGrid: self.listSurfPointNegative, self.listSurfPointPositive = listSurfPoint[self.name][:nNegPoints-1],listSurfPoint[self.name][nNegPoints:]
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
#
#
#
class ExternalInput(InputData.ParameterInput):
  """
    Class for reading in the External post processor.
  """

ExternalInput.createClass("PostProcessor", False, baseNode=ModelInput)
EMethodInput = InputData.parameterInputFactory("method")
ExternalInput.addSub(EMethodInput)
EFunctionInput = InputData.parameterInputFactory("Function", contentType=InputData.StringType)
EFunctionInput.addParam("class", InputData.StringType)
EFunctionInput.addParam("type", InputData.StringType)
ExternalInput.addSub(EFunctionInput)



class ExternalPostProcessor(BasePostProcessor):
  """
    ExternalPostProcessor class. It will apply an arbitrary python function to
    a dataset and append each specified function's output to the output data
    object, thus the function should produce a scalar value per row of data. I
    have no idea what happens if the function produces multiple outputs.
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self, messageHandler)
    self.methodsToRun = []              # A list of strings specifying what
                                        # methods the user wants to compute from
                                        # the external interfaces

    self.externalInterfaces = set()     # A set of Function objects that
                                        # hopefully contain definitions for all
                                        # of the methods the user wants

    self.printTag = 'POSTPROCESSOR EXTERNAL FUNCTION'
    self.requiredAssObject = (True, (['Function'], ['n']))

  def inputToInternal(self, currentInput):
    """
      Function to convert the received input into a format this object can
      understand
      @ In, currentInput, dataObjects or list, Some form of data object or list
        of data objects handed to the post-processor
      @ Out, inputDict, dict, An input dictionary this object can process
    """

    if type(currentInput) == dict and 'targets' in currentInput.keys():
      return

    if type(currentInput) != list:
      currentInput = [currentInput]

    inputDict = {'targets':{}, 'metadata':{}}
    metadata = []
    for item in currentInput:
      inType = None
      if hasattr(item, 'type'):
        inType = item.type
      elif type(item) in [list]:
        inType = "list"

      if isinstance(item,Files.File):
        if currentInput.subtype == 'csv':
          self.raiseAWarning(self, 'Input type ' + inType + ' not yet implemented. I am going to skip it.')
      elif inType == 'HDF5':
        # TODO
        self.raiseAWarning(self, 'Input type ' + inType + ' not yet implemented. I am going to skip it.')
      elif inType == 'PointSet':
        for param in item.getParaKeys('input'):
          inputDict['targets'][param] = item.getParam('input', param)
        for param in item.getParaKeys('output'):
          inputDict['targets'][param] = item.getParam('output', param)
        metadata.append(item.getAllMetadata())
      elif inType =='HistorySet':
        outs, ins = item.getOutParametersValues(nodeId = 'ending'), item.getInpParametersValues(nodeId = 'ending')
        for param in item.getParaKeys('output'):
          inputDict['targets'][param] = [value[param] for value in outs.values()]
        for param in item.getParaKeys('input'):
          inputDict['targets'][param] =  [value[param] for value in ins.values()]
        metadata.append(item.getAllMetadata())
      elif inType != 'list':
        self.raiseAWarning(self, 'Input type ' + type(item).__name__ + ' not recognized. I am going to skip it.')

      # Not sure if we need it, but keep a copy of every inputs metadata
      inputDict['metadata'] = metadata

    if len(inputDict['targets'].keys()) == 0:
      self.raiseAnError(IOError, 'No input variables have been found in the input objects!')

    for interface in self.externalInterfaces:
      for _ in self.methodsToRun:
        # The function should reference self and use the same variable names
        # as the xml file
        for param in interface.parameterNames():
          if param not in inputDict['targets']:
            self.raiseAnError(IOError, self, 'variable \"' + param
                                             + '\" unknown. Please verify your '
                                             + 'external script ('
                                             + interface.functionFile
                                             + ') variables match the data'
                                             + ' available in your dataset.')
    return inputDict

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the External pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = runInfo['WorkingDir']
    for key in self.assemblerDict.keys():
      if 'Function' in key:
        for val in self.assemblerDict[key]:
          self.externalInterfaces.add(val[3])

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this
      specialized class and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = ExternalInput()
    paramInput.parseNode(xmlNode)
    for child in paramInput.subparts:
      if child.getName() == 'method':
        methods = child.value.split(',')
        self.methodsToRun.extend(methods)

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler
        object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed
        results
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1:
      # #TODO This does not feel right
      self.raiseAnError(RuntimeError, 'No available Output to collect (Run probably did not finish yet)')
    dataLenghtHistory = {}
    inputList = finishedJob.getEvaluation()[0]
    outputDict = finishedJob.getEvaluation()[1]

    if isinstance(output,Files.File):
      self.raiseAWarning('Output type File not yet implemented. I am going to skip it.')
    elif output.type == 'DataObjects':
      self.raiseAWarning('Output type ' + type(output).__name__
                         + ' not yet implemented. I am going to skip it.')
    elif output.type == 'HDF5':
      self.raiseAWarning('Output type ' + type(output).__name__
                         + ' not yet implemented. I am going to skip it.')
    elif output.type in ['PointSet','HistorySet'] :
      requestedInput = output.getParaKeys('input')
      ## If you want to be able to dynamically add columns to your data, then
      ## you should use this commented line, otherwise only the information
      ## asked for by the user in the output data object will be available

      # requestedOutput = list(set(output.getParaKeys('output') + self.methodsToRun))
      requestedOutput = output.getParaKeys('output')

      ## The user can simply ask for a computation that may exist in multiple
      ## interfaces, in that case, we will need to qualify their names for the
      ## output. The names should already be qualified from the outputDict.
      ## However, the user may have already qualified the name, so make sure and
      ## test whether the unqualified name exists in the requestedOutput before
      ## replacing it.
      for key, replacements in outputDict['qualifiedNames'].iteritems():
        if key in requestedOutput:
          requestedOutput.remove(key)
          requestedOutput.extend(replacements)

      ## Grab all data from the outputDict and anything else requested not
      ## present in the outputDict will be copied from the input data.
      ## TODO: User may want to specify which dataset the parameter comes from.
      ##       For now, we assume that if we find more than one an error will
      ##       occur.
      ## FIXME: There is an issue that the data size should be determined before
      ##        entering this loop, otherwise if say a scalar is first added,
      ##        then dataLength will be 1 and everything longer will be placed
      ##        in the Metadata.
      ##        How do we know what size the output data should be?
      dataLength = None
      for key in requestedInput + requestedOutput:
        storeInOutput = True
        value = []
        if key in outputDict:
          value = outputDict[key]
        else:
          foundCount = 0
          if key in requestedInput:
            for inputData in inputList:
              if key in inputData.getParametersValues('input',nodeId = 'ending').keys() if inputData.type == 'PointSet' else inputData.getParametersValues('input',nodeId = 'ending').values()[-1].keys():
                if inputData.type == 'PointSet':
                  value = inputData.getParametersValues('input',nodeId = 'ending')[key]
                else:
                  value = [value[key] for value in inputData.getParametersValues('input',nodeId = 'ending').values()]
                foundCount += 1
          else:
            for inputData in inputList:
              if key in inputData.getParametersValues('output',nodeId = 'ending').keys() if inputData.type == 'PointSet' else inputData.getParametersValues('output',nodeId = 'ending').values()[-1].keys():
                if inputData.type == 'PointSet':
                  value = inputData.getParametersValues('output',nodeId = 'ending')[key]
                else:
                  value = [value[key] for value in inputData.getParametersValues('output',nodeId = 'ending').values()]
                foundCount += 1

          if foundCount == 0:
            self.raiseAnError(IOError, key + ' not found in the input '
                                            + 'object or the computed output '
                                            + 'object.')
          elif foundCount > 1:
            self.raiseAnError(IOError, key + ' is ambiguous since it occurs'
                                            + ' in multiple input objects.')

        ## We need the size to ensure the data size is consistent, but there
        ## is no guarantee the data is not scalar, so this check is necessary
        myLength = 1
        if not hasattr(value, "__iter__"):
          value = [value]
        myLength = len(value)

        if dataLength is None:
          dataLength = myLength
        elif dataLength != myLength:
          self.raiseAWarning('Requested output for ' + key + ' has a'
                                    + ' non-conformant data size ('
                                    + str(dataLength) + ' vs ' + str(myLength)
                                    + '), it is being placed in the metadata.')
          storeInOutput = False

        ## Finally, no matter what, place the requested data somewhere
        ## accessible
        if storeInOutput:
          if key in requestedInput:
            for histNum, val in enumerate(value):
              param = key if output.type == 'PointSet' else [histNum+1,key]
              output.updateInputValue(param, val)
          else:
            for histNum, val in enumerate(value):
              if output.type == 'HistorySet':
                if histNum+1 in dataLenghtHistory.keys():
                  if dataLenghtHistory[histNum+1] != len(val): self.raiseAnError(IOError, key + ' the size of the arrays for history '+str(histNum+1)+' are different!')
                else: dataLenghtHistory[histNum+1] = len(val)
              param = key if output.type == 'PointSet' else [histNum+1,key]
              output.updateOutputValue(param, val)
        else:
          if not hasattr(value, "__iter__"):
            value = [value]
          for val in value:
            output.updateMetadata(key, val)
    else:
      self.raiseAnError(IOError, 'Unknown output type: ' + str(output.type))

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it performs
      the action defined in the external pp
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDict, dict, Dictionary containing the post-processed results
    """
    input = self.inputToInternal(inputIn)
    outputDict = {'qualifiedNames' : {}}
    ## This will map the name to its appropriate interface and method
    ## in the case of a function being defined in two separate files, we
    ## qualify the output by appending the name of the interface from which it
    ## originates
    methodMap = {}

    ## First check all the requested methods are available and if there are
    ## duplicates then qualify their names for the user
    for method in self.methodsToRun:
      matchingInterfaces = []
      for interface in self.externalInterfaces:
        if method in interface.availableMethods():
          matchingInterfaces.append(interface)
      if len(matchingInterfaces) == 0:
        self.raiseAWarning(method + ' not found. I will skip it.')
      elif len(matchingInterfaces) == 1:
        methodMap[method] = (matchingInterfaces[0], method)
      else:
        outputDict['qualifiedNames'][method] = []
        for interface in matchingInterfaces:
          methodName = interface.name + '.' + method
          methodMap[methodName] = (interface, method)
          outputDict['qualifiedNames'][method].append(methodName)

    ## Evaluate the method and add it to the outputDict, also if the method
    ## adjusts the input data, then you should update it as well.
    warningMessages = []
    for methodName, (interface, method) in methodMap.iteritems():
      outputDict[methodName] = interface.evaluate(method, input['targets'])
      if outputDict[methodName] is None: self.raiseAnError(Exception,"the method "+methodName+" has not produced any result. It needs to return a result!")
      for target in input['targets']:
        if hasattr(interface, target):
          #if target not in outputDict.keys():
          if target not in methodMap.keys():
            attributeInSelf = getattr(interface, target)
            if len(np.atleast_1d(attributeInSelf)) != len(np.atleast_1d(input['targets'][target])) or (np.atleast_1d(attributeInSelf) - np.atleast_1d(input['targets'][target])).all():
              if target in outputDict.keys(): self.raiseAWarning("In Post-Processor "+ self.name +" the modified variable "+target+
                               " has the same name of a one already modified throuhg another Function method." +
                               " This method overwrites the input DataObject variable value")
              outputDict[target] = attributeInSelf
          else:
            warningMessages.append("In Post-Processor "+ self.name +" the method "+method+
                               " has the same name of a variable contained in the input DataObject." +
                               " This method overwrites the input DataObject variable value")
    for msg in list(set(warningMessages)): self.raiseAWarning(msg)

    for target in input['targets'].keys():
      if target not in outputDict.keys() and target in input['targets'].keys():
        outputDict[target] = input['targets'][target]

    return outputDict

#
#
#

class TopologicalDecompositionInput(InputData.ParameterInput):
  """
    class for reading in the topological decomposition block
  """

TopologicalDecompositionInput.createClass("PostProcessor", False, baseNode=ModelInput)
TDGraphInput = InputData.parameterInputFactory("graph", contentType=InputData.StringType)
TopologicalDecompositionInput.addSub(TDGraphInput)
TDGradientInput = InputData.parameterInputFactory("gradient", contentType=InputData.StringType)
TopologicalDecompositionInput.addSub(TDGradientInput)
TDBetaInput = InputData.parameterInputFactory("beta", contentType=InputData.FloatType)
TopologicalDecompositionInput.addSub(TDBetaInput)
TDKNNInput = InputData.parameterInputFactory("knn", contentType=InputData.IntegerType)
TopologicalDecompositionInput.addSub(TDKNNInput)
TDWeightedInput = InputData.parameterInputFactory("weighted", contentType=InputData.StringType) #bool
TopologicalDecompositionInput.addSub(TDWeightedInput)
TDPersistenceInput = InputData.parameterInputFactory("persistence", contentType=InputData.StringType)
TopologicalDecompositionInput.addSub(TDPersistenceInput)
TDSimplificationInput = InputData.parameterInputFactory("simplification", contentType=InputData.FloatType)
TopologicalDecompositionInput.addSub(TDSimplificationInput)
TDParametersInput = InputData.parameterInputFactory("parameters", contentType=InputData.StringType)
TopologicalDecompositionInput.addSub(TDParametersInput)
TDResponseInput = InputData.parameterInputFactory("response", contentType=InputData.StringType)
TopologicalDecompositionInput.addSub(TDResponseInput)
TDNormalizationInput = InputData.parameterInputFactory("normalization", contentType=InputData.StringType)
TopologicalDecompositionInput.addSub(TDNormalizationInput)

#
class TopologicalDecomposition(BasePostProcessor):
  """
    TopologicalDecomposition class - Computes an approximated hierarchical
    Morse-Smale decomposition from an input point cloud consisting of an
    arbitrary number of input parameters and a response value per input point
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self, messageHandler)
    self.acceptedGraphParam = ['approximate knn', 'delaunay', 'beta skeleton', \
                               'relaxed beta skeleton']
    self.acceptedPersistenceParam = ['difference','probability','count']#,'area']
    self.acceptedGradientParam = ['steepest', 'maxflow']
    self.acceptedNormalizationParam = ['feature', 'zscore', 'none']

    # Some default arguments
    self.gradient = 'steepest'
    self.graph = 'beta skeleton'
    self.beta = 1
    self.knn = -1
    self.simplification = 0
    self.persistence = 'difference'
    self.normalization = None
    self.weighted = False
    self.parameters = {}

  def inputToInternal(self, currentInp):
    """
      Function to convert the incoming input into a usable format
      @ In, currentInp, list or DataObjects, The input object to process
      @ Out, inputDict, dict, the converted input
    """
    if type(currentInp) == list  : currentInput = currentInp [-1]
    else                         : currentInput = currentInp
    if type(currentInput) == dict:
      if 'features' in currentInput.keys(): return currentInput
    inputDict = {'features':{}, 'targets':{}, 'metadata':{}}
    if hasattr(currentInput, 'type'):
      inType = currentInput.type
    elif type(currentInput).__name__ == 'list':
      inType = 'list'
    else:
      self.raiseAnError(IOError, self.__class__.__name__,
                        ' postprocessor accepts files, HDF5, Data(s) only. ',
                        ' Requested: ', type(currentInput))

    if inType not in ['HDF5', 'PointSet', 'list'] and not isinstance(currentInput,Files.File):
      self.raiseAnError(IOError, self, self.__class__.__name__ + ' post-processor only accepts files, HDF5, or DataObjects! Got ' + str(inType) + '!!!!')
    # FIXME: implement this feature
    if isinstance(currentInput,Files.File):
      if currentInput.subtype == 'csv': pass
    # FIXME: implement this feature
    if inType == 'HDF5': pass  # to be implemented
    if inType in ['PointSet']:
      for targetP in self.parameters['features']:
        if   targetP in currentInput.getParaKeys('input'):
          inputDict['features'][targetP] = currentInput.getParam('input' , targetP)
        elif targetP in currentInput.getParaKeys('output'):
          inputDict['features'][targetP] = currentInput.getParam('output', targetP)
      for targetP in self.parameters['targets']:
        if   targetP in currentInput.getParaKeys('input'):
          inputDict['targets'][targetP] = currentInput.getParam('input' , targetP)
        elif targetP in currentInput.getParaKeys('output'):
          inputDict['targets'][targetP] = currentInput.getParam('output', targetP)
      inputDict['metadata'] = currentInput.getAllMetadata()
    # now we check if the sampler that genereted the samples are from adaptive... in case... create the grid
    if 'SamplerType' in inputDict['metadata'].keys(): pass
    return inputDict

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = TopologicalDecompositionInput()
    paramInput.parseNode(xmlNode)
    for child in paramInput.subparts:
      if child.getName() == "graph":
        self.graph = child.value.encode('ascii').lower()
        if self.graph not in self.acceptedGraphParam:
          self.raiseAnError(IOError, 'Requested unknown graph type: ',
                            self.graph, '. Available options: ',
                            self.acceptedGraphParam)
      elif child.getName() == "gradient":
        self.gradient = child.value.encode('ascii').lower()
        if self.gradient not in self.acceptedGradientParam:
          self.raiseAnError(IOError, 'Requested unknown gradient method: ',
                            self.gradient, '. Available options: ',
                            self.acceptedGradientParam)
      elif child.getName() == "beta":
        self.beta = child.value
        if self.beta <= 0 or self.beta > 2:
          self.raiseAnError(IOError, 'Requested invalid beta value: ',
                            self.beta, '. Allowable range: (0,2]')
      elif child.getName() == 'knn':
        self.knn = child.value
      elif child.getName() == 'simplification':
        self.simplification = child.value
      elif child.getName() == 'persistence':
        self.persistence = child.value.encode('ascii').lower()
        if self.persistence not in self.acceptedPersistenceParam:
          self.raiseAnError(IOError, 'Requested unknown persistence method: ',
                            self.persistence, '. Available options: ',
                            self.acceptedPersistenceParam)
      elif child.getName() == 'parameters':
        self.parameters['features'] = child.value.strip().split(',')
        for i, parameter in enumerate(self.parameters['features']):
          self.parameters['features'][i] = self.parameters['features'][i].encode('ascii')
      elif child.getName() == 'weighted':
        self.weighted = child.value in ['True', 'true']
      elif child.getName() == 'response':
        self.parameters['targets'] = child.value
      elif child.getName() == 'normalization':
        self.normalization = child.value.encode('ascii').lower()
        if self.normalization not in self.acceptedNormalizationParam:
          self.raiseAnError(IOError, 'Requested unknown normalization type: ',
                            self.normalization, '. Available options: ',
                            self.acceptedNormalizationParam)

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1:
      # TODO This does not feel right
      self.raiseAnError(RuntimeError,'No available output to collect (run probably did not finish yet)')
    inputList = finishedJob.getEvaluation()[0]
    outputDict = finishedJob.getEvaluation()[1]

    if type(output).__name__ in ["str", "unicode", "bytes"]:
      self.raiseAWarning('Output type ' + type(output).__name__ + ' not'
                         + ' yet implemented. I am going to skip it.')
    elif output.type == 'Datas':
      self.raiseAWarning('Output type ' + type(output).__name__ + ' not'
                         + ' yet implemented. I am going to skip it.')
    elif output.type == 'HDF5':
      self.raiseAWarning('Output type ' + type(output).__name__ + ' not'
                         + ' yet implemented. I am going to skip it.')
    elif output.type == 'PointSet':
      requestedInput = output.getParaKeys('input')
      requestedOutput = output.getParaKeys('output')
      dataLength = None
      for inputData in inputList:
        # Pass inputs from input data to output data
        for key, value in inputData.getParametersValues('input').items():
          if key in requestedInput:
            # We need the size to ensure the data size is consistent, but there
            # is no guarantee the data is not scalar, so this check is necessary
            myLength = 1
            if hasattr(value, "__len__"):
              myLength = len(value)

            if dataLength is None:
              dataLength = myLength
            elif dataLength != myLength:
              dataLength = max(dataLength, myLength)
              self.raiseAWarning('Data size is inconsistent. Currently set to '
                                 + str(dataLength) + '.')

            for val in value:
              output.updateInputValue(key, val)

        # Pass outputs from input data to output data
        for key, value in inputData.getParametersValues('output').items():
          if key in requestedOutput:
            # We need the size to ensure the data size is consistent, but there
            # is no guarantee the data is not scalar, so this check is necessary
            myLength = 1
            if hasattr(value, "__len__"):
              myLength = len(value)

            if dataLength is None:
              dataLength = myLength
            elif dataLength != myLength:
              dataLength = max(dataLength, myLength)
              self.raiseAWarning('Data size is inconsistent. Currently set to '
                                      + str(dataLength) + '.')

            for val in value:
              output.updateOutputValue(key, val)

        # Append the min/max labels to the data whether the user wants them or
        # not, and place the hierarchy information into the metadata
        for key, value in outputDict.iteritems():
          if key in ['minLabel', 'maxLabel']:
            output.updateOutputValue(key, [value])
          elif key in ['hierarchy']:
            output.updateMetadata(key, [value])
    else:
      self.raiseAnError(IOError,'Unknown output type:',output.type)

  def userInteraction(self):
    """
      A placeholder for allowing user's to interact and tweak the model in-situ
      before saving the analysis results
      @ In, None
      @ Out, None
    """
    pass

  def run(self, inputIn):
    """
      Function to finalize the filter => execute the filtering
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDict, dict, Dictionary containing the post-processed results
    """

    internalInput = self.inputToInternal(inputIn)
    outputDict = {}

    myDataIn = internalInput['features']
    myDataOut = internalInput['targets']

    self.outputData = myDataOut[self.parameters['targets'].encode('UTF-8')]
    self.pointCount = len(self.outputData)
    self.dimensionCount = len(self.parameters['features'])

    self.inputData = np.zeros((self.pointCount, self.dimensionCount))
    for i, lbl in enumerate(self.parameters['features']):
      self.inputData[:, i] = myDataIn[lbl.encode('UTF-8')]

    if self.weighted:
      self.weights = inputIn[0].getMetadata('PointProbability')
    else:
      self.weights = None

    self.names = self.parameters['features'] + [self.parameters['targets']]

    self.__amsc = None
    self.userInteraction()

    ## Possibly load this here in case people have trouble building it, so it
    ## only errors if they try to use it?
    from AMSC_Object import AMSC_Object

    if self.__amsc is None:
      self.__amsc = AMSC_Object(X=self.inputData, Y=self.outputData,
                                w=self.weights, names=self.names,
                                graph=self.graph, gradient=self.gradient,
                                knn=self.knn, beta=self.beta,
                                normalization=self.normalization,
                                persistence=self.persistence, debug=False)

    self.__amsc.Persistence(self.simplification)
    partitions = self.__amsc.Partitions()

    outputDict['minLabel'] = np.zeros(self.pointCount)
    outputDict['maxLabel'] = np.zeros(self.pointCount)
    for extPair, indices in partitions.iteritems():
      for idx in indices:
        outputDict['minLabel'][idx] = extPair[0]
        outputDict['maxLabel'][idx] = extPair[1]
    outputDict['hierarchy'] = self.__amsc.PrintHierarchy()
    self.__amsc.BuildModels()
    linearFits = self.__amsc.SegmentFitCoefficients()
    linearFitnesses = self.__amsc.SegmentFitnesses()

    for key in linearFits.keys():
      coefficients = linearFits[key]
      rSquared = linearFitnesses[key]
      outputDict['coefficients_%d_%d' % (key[0], key[1])] = coefficients
      outputDict['R2_%d_%d' % (key[0], key[1])] = rSquared

    return outputDict

try:
  import PySide.QtCore as qtc
  class QTopologicalDecomposition(TopologicalDecomposition,qtc.QObject):
    """
      TopologicalDecomposition class - Computes an approximated hierarchical
      Morse-Smale decomposition from an input point cloud consisting of an
      arbitrary number of input parameters and a response value per input point
    """
    requestUI = qtc.Signal(str,str,dict)
    def __init__(self, messageHandler):
      """
       Constructor
       @ In, messageHandler, message handler object
       @ Out, None
      """

      TopologicalDecomposition.__init__(self, messageHandler)
      qtc.QObject.__init__(self)

      self.interactive = False
      self.uiDone = True ## If it has not been requested, then we are not waiting for a UI

    def _localWhatDoINeed(self):
      """
        This method is a local mirror of the general whatDoINeed method.
        It is implemented by the samplers that need to request special objects
        @ In , None, None
        @ Out, needDict, list of objects needed
      """
      return {'internal':[(None,'app')]}

    def _localGenerateAssembler(self,initDict):
      """
        Generates the assembler.
        @ In, initDict, dict of init objects
        @ Out, None
      """
      self.app = initDict['internal']['app']
      if self.app is None:
        self.interactive = False

    def _localReadMoreXML(self, xmlNode):
      """
        Function to grab the names of the methods this post-processor will be
        using
        @ In, xmlNode    : Xml element node
        @ Out, None
      """
      TopologicalDecomposition._localReadMoreXML(self, xmlNode)
      for child in xmlNode:
        if child.tag == 'interactive':
          self.interactive = True

    def userInteraction(self):
      """
        Launches an interface allowing the user to tweak specific model
        parameters before saving the results to the output object(s).
        @ In, None
        @ Out, None
      """
      self.uiDone = not self.interactive

      if self.interactive:
        ## Connect our own signal to the slot on the main thread
        self.requestUI.connect(self.app.createUI)

        ## Connect our own slot to listen for whenver the main thread signals a
        ## window has been closed
        self.app.windowClosed.connect(self.signalDone)

        ## Give this UI a unique id in case other threads are requesting UI
        ##  elements
        uiID = unicode(id(self))

        ## Send the request for a UI thread to the main application
        self.requestUI.emit('TopologyWindow', uiID,
                            {'X':self.inputData, 'Y':self.outputData,
                             'w':self.weights, 'names':self.names,
                             'graph':self.graph, 'gradient': self.gradient,
                             'knn':self.knn, 'beta':self.beta,
                             'normalization':self.normalization, 'debug':False})

        ## Spinlock will wait until this instance's window has been closed
        while(not self.uiDone):
          time.sleep(1)

        ## First check that the requested UI exists, and then if that UI has the
        ## requested information, if not proceed as if it were not an
        ## interactive session.
        if uiID in self.app.UIs and hasattr(self.app.UIs[uiID],'amsc'):
          self.__amsc = self.app.UIs[uiID].amsc
          self.simplification = self.app.UIs[uiID].amsc.Persistence()
        else:
          self.__amsc = None

    def signalDone(self,uiID):
      """
        In Qt language, this is a slot that will accept a signal from the UI
        saying that it has completed, thus allowing the computation to begin
        again with information updated by the user in the UI.
        @In, uiID, string, the ID of the user interface that signaled its
            completion. Thus, if several UI windows are open, we don't proceed,
            until the correct one has signaled it is done.
        @Out, None
      """
      if uiID == unicode(id(self)):
        self.uiDone = True
except ImportError as e:
  pass
#
#
#
#
class DataMining(BasePostProcessor):
  """
    DataMiningPostProcessor class. It will apply the specified KDD algorithms in
    the models to a dataset, each specified algorithm's output can be loaded to
    dataObject.
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self, messageHandler)
    self.printTag = 'POSTPROCESSOR DATAMINING'

    self.requiredAssObject = (True, (['PreProcessor','Metric'], ['-1','-1']))

    self.solutionExport = None                            ## A data object to
                                                          ## hold derived info
                                                          ## about the algorithm
                                                          ## being performed,
                                                          ## e.g., cluster
                                                          ## centers or a
                                                          ## projection matrix
                                                          ## for dimensionality
                                                          ## reduction methods

    self.labelFeature = None                              ## User customizable
                                                          ## column name for the
                                                          ## labels associated
                                                          ## to a clustering or
                                                          ## a DR algorithm

    self.PreProcessor = None
    self.metric = None

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method.
      It is implemented by the samplers that need to request special objects
      @ In , None, None
      @ Out, dict, dictionary of objects needed
    """
    return {'internal':[(None,'jobHandler')]}

  def _localGenerateAssembler(self,initDict):
    """Generates the assembler.
      @ In, initDict, dict, init objects
      @ Out, None
    """
    self.jobHandler = initDict['internal']['jobHandler']

  def inputToInternal(self, currentInp):
    """
      Function to convert the received input into a format this object can
      understand
      @ In, currentInp, list or DataObjects, Some form of data object or list of
        data objects handed to the post-processor
      @ Out, inputDict, dict, An input dictionary this object can process
    """

    if type(currentInp) == list:
      currentInput = currentInp[-1]
    else:
      currentInput = currentInp

    if currentInput.type == 'HistorySet' and self.PreProcessor is None and self.metric is None: # for testing time dependent dm - time dependent clustering
      inputDict = {'Features':{}, 'parameters':{}, 'Labels':{}, 'metadata':{}}

      # FIXME, this needs to be changed for asynchronous HistorySet
      if self.pivotParameter in currentInput.getParam('output',1).keys():
        self.pivotVariable = currentInput.getParam('output',1)[self.pivotParameter]
      else:
        self.raiseAnError(ValueError, 'Pivot variable not found in input historyset')
      # end of FIXME

      historyKey = currentInput.getOutParametersValues().keys()
      numberOfSample = len(historyKey)
      numberOfHistoryStep = len(self.pivotVariable)

      if self.initializationOptionDict['KDD']['Features'] == 'input':
        self.raiseAnError(ValueError, 'To perform data mining over input please use SciKitLearn library')
      elif self.initializationOptionDict['KDD']['Features'] in ['output', 'all']:
        features = currentInput.getParaKeys('output')
        features.remove(self.pivotParameter)
      else:
        features = self.initializationOptionDict['KDD']['Features'].split(',')

      for param in features:
        inputDict['Features'][param] = np.zeros(shape=(numberOfSample,numberOfHistoryStep))
        for cnt, keyH in enumerate(historyKey):
          inputDict['Features'][param][cnt,:] = currentInput.getParam('output', keyH)[param]

      inputDict['metadata'] = currentInput.getAllMetadata()
      return inputDict

    if type(currentInp) == dict:
      if 'Features' in currentInput.keys(): return
    if isinstance(currentInp, Files.File):
      if currentInput.subtype == 'csv':
        self.raiseAnError(IOError, 'CSV File received as an input!')
    if currentInput.type == 'HDF5':
      self.raiseAnError(IOError, 'HDF5 Object received as an input!')

    if self.PreProcessor != None:
      inputDict = {'Features':{}, 'parameters':{}, 'Labels':{}, 'metadata':{}}
      if self.initializationOptionDict['KDD']['Features'] == 'input':
        features = currentInput.getParaKeys('input')
      elif self.initializationOptionDict['KDD']['Features'] == 'output':
        features = currentInput.getParaKeys('output')
      else:
        features = self.initializationOptionDict['KDD']['Features'].split(',')

      tempData = self.PreProcessor.interface.inputToInternal(currentInp)

      preProcessedData = self.PreProcessor.interface.run(tempData)
      if self.initializationOptionDict['KDD']['Features'] == 'input':
        inputDict['Features'] = copy.deepcopy(preProcessedData['data']['input'])
      elif self.initializationOptionDict['KDD']['Features'] == 'output':
        inputDict['Features'] = copy.deepcopy(preProcessedData['data']['output'])
      else:
        features = self.initializationOptionDict['KDD']['Features'].split(',')
        for param in currentInput.getParaKeys('input'):
          if param in features:
            inputDict['Features'][param] = copy.deepcopy(preProcessedData['data']['input'][param])
        for param in currentInput.getParaKeys('output'):
          if param in features:
            inputDict['Features'][param] = copy.deepcopy(preProcessedData['data']['output'][param])

      inputDict['metadata'] = currentInput.getAllMetadata()

      return inputDict

    inputDict = {'Features':{}, 'parameters':{}, 'Labels':{}, 'metadata':{}}

    if currentInput.type in ['PointSet']:
      ## Get what is available in the data object being operated on
      ## This is potentially more information than we need at the moment, but
      ## it will make the code below easier to read and highlights where objects
      ## are reused more readily
      allInputFeatures = currentInput.getParaKeys('input')
      allOutputFeatures = currentInput.getParaKeys('output')
      if self.initializationOptionDict['KDD']['Features'] == 'input':
        for param in allInputFeatures:
          inputDict['Features'][param] = currentInput.getParam('input', param)
      elif self.initializationOptionDict['KDD']['Features'] == 'output':
        for param in allOutputFeatures:
          inputDict['Features'][param] = currentInput.getParam('output', param)
      elif self.initializationOptionDict['KDD']['Features'] == 'all':
        for param in allInputFeatures:
          inputDict['Features'][param] = currentInput.getParam('input', param)
        for param in allOutputFeatures:
          inputDict['Features'][param] = currentInput.getParam('output', param)
      else:
        ## Get what the user asks requests
        features = set(self.initializationOptionDict['KDD']['Features'].split(','))

        ## Now intersect what the user wants and what is available.
        ## NB: this will not error, if the user asks for something that does not
        ##     exist in the data, it will silently ignore it.
        inParams = list(features.intersection(allInputFeatures))
        outParams = list(features.intersection(allOutputFeatures))

        for param in inParams:
          inputDict['Features'][param] = currentInput.getParam('input', param)
        for param in outParams:
          inputDict['Features'][param] = currentInput.getParam('output', param)

    elif currentInput.type in ['HistorySet']:
      if self.initializationOptionDict['KDD']['Features'] == 'input':
        for param in currentInput.getParaKeys('input'):
          inputDict['Features'][param] = currentInput.getParam('input', param)
      elif self.initializationOptionDict['KDD']['Features'] == 'output':
        inputDict['Features'] = currentInput.getOutParametersValues()
      elif self.initializationOptionDict['KDD']['Features'] == 'all':
        for param in allInputFeatures:
          inputDict['Features'][param] = currentInput.getParam('input', param)
        for param in allOutputFeatures:
          inputDict['Features'][param] = currentInput.getParam('output', param)
      else:
        features = set(self.initializationOptionDict['KDD']['Features'].split(','))
        allInputFeatures = currentInput.getParaKeys('input')
        allOutputFeatures = currentInput.getParaKeys('output')
        inParams = list(features.intersection(allInputFeatures))
        outParams = list(features.intersection(allOutputFeatures))
        inputDict['Features'] = {}
        for hist in currentInput._dataContainer['outputs'].keys():
          inputDict['Features'][hist] = {}
          for param in inParams:
            inputDict['Features'][hist][param] = currentInput._dataContainer['inputs'][hist][param]
          for param in outParams:
            inputDict['Features'][hist][param] = currentInput._dataContainer['outputs'][hist][param]

      inputDict['metadata'] = currentInput.getAllMetadata()

    ## Redundant if-conditional preserved as a placeholder for potential future
    ## development working directly with files
    # elif isinstance(currentInp, Files.File):
    #   self.raiseAnError(IOError, 'Unsupported input type (' + currentInput.subtype + ') for PostProcessor ' + self.name + ' must be a PointSet.')
    else:
      self.raiseAnError(IOError, 'Unsupported input type (' + currentInput.type + ') for PostProcessor ' + self.name + ' must be a PointSet.')
    return inputDict

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the DataMining pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """

    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    self.__workingDir = runInfo['WorkingDir']

    if "SolutionExport" in initDict:
      self.solutionExport = initDict["SolutionExport"]

    if "PreProcessor" in self.assemblerDict:
      self.PreProcessor = self.assemblerDict['PreProcessor'][0][3]
      if not '_inverse' in dir(self.PreProcessor.interface):
        self.raiseAnError(IOError, 'PostProcessor ' + self.name + ' is using a pre-processor where the method inverse has not implemented')


    if 'Metric' in self.assemblerDict:
      self.metric = self.assemblerDict['Metric'][0][3]

  def _localReadMoreXML(self, xmlNode):
    """
      Function that reads the portion of the xml input that belongs to this specialized class
      and initializes some elements based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    ## By default, we want to name the 'labels' by the name of this
    ## postprocessor, but that name is not available before processing the XML
    ## At this point, we have that information
    self.initializationOptionDict = {}

    for child in xmlNode:
      if child.tag == 'KDD':
        if child.attrib:
          ## I'm not sure what this thing is used for, but it seems to make more
          ## sense to only put data that is not otherwise handled rather than
          ## put all of the information and then to remove the ones we process.
          ## - dpm 6/8/16
          self.initializationOptionDict[child.tag] = {}
          for key,value in child.attrib.iteritems():
            if key == 'lib':
              self.type = value
            elif key == 'labelFeature':
              self.labelFeature = value
            else:
              self.initializationOptionDict[child.tag][key] = value
        else:
          self.initializationOptionDict[child.tag] = utils.tryParse(child.text)

        for childChild in child:
          if childChild.attrib and not childChild.tag == 'PreProcessor':
            self.initializationOptionDict[child.tag][childChild.tag] = dict(childChild.attrib)
          else:
            self.initializationOptionDict[child.tag][childChild.tag] = utils.tryParse(childChild.text)
      elif child.tag == 'pivotParameter':
        self.pivotParameter = child.text

    if not hasattr(self, 'pivotParameter'):
      #TODO, if doing time dependent data mining that needs this, an error
      # should be thrown
      self.pivotParameter = None

    if self.type:
      #TODO unSurpervisedEngine needs to be able to handle both methods
      # without this if statement.
      if self.pivotParameter is not None:
        self.unSupervisedEngine = unSupervisedLearning.returnInstance("temporalSciKitLearn", self, **self.initializationOptionDict['KDD'])
      else:
        self.unSupervisedEngine = unSupervisedLearning.returnInstance(self.type, self, **self.initializationOptionDict['KDD'])
    else:
      self.raiseAnError(IOError, 'No Data Mining Algorithm is supplied!')

    ## If the user has not defined a label feature, then we will force it to be
    ## named by the PostProcessor name followed by:
    ## the word 'Labels' for clustering/GMM models;
    ## the word 'Dimension' + a numeric id for dimensionality reduction
    ## algorithms
    if self.labelFeature is None:
      if self.unSupervisedEngine.getDataMiningType() in ['cluster','mixture']:
        self.labelFeature = self.name+'Labels'
      elif self.unSupervisedEngine.getDataMiningType() in ['decomposition','manifold']:
        self.labelFeature = self.name+'Dimension'

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler
        object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed
        results
      @ Out, None
    """
    ## When does this actually happen?
    if finishedJob.getEvaluation() == -1:
      self.raiseAnError(RuntimeError, 'No available Output to collect (Run probably is not finished yet)')
    dataMineDict = finishedJob.getEvaluation()[1]
    for key in dataMineDict['outputs']:
      for param in output.getParaKeys('output'):
        if key == param:
          output.removeOutputValue(key)
      if output.type == 'PointSet':
        for value in dataMineDict['outputs'][key]:
          output.updateOutputValue(key, copy.copy(value))
      elif output.type == 'HistorySet':
        if self.PreProcessor is not None or self.metric is not None:
          for index,value in np.ndenumerate(dataMineDict['outputs'][key]):
            firstHist = output._dataContainer['outputs'].keys()[0]
            firstVar  = output._dataContainer['outputs'][index[0]+1].keys()[0]
            timeLength = output._dataContainer['outputs'][index[0]+1][firstVar].size
            arrayBase = value * np.ones(timeLength)
            output.updateOutputValue([index[0]+1,key], arrayBase)
        else:
          tlDict = finishedJob.getEvaluation()[1]
          historyKey = output.getOutParametersValues().keys()
          for index, keyH in enumerate(historyKey):
            for keyL in tlDict['outputs'].keys():
              output.updateOutputValue([keyH,keyL], tlDict['outputs'][keyL][index,:])

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    Input = self.inputToInternal(inputIn)
    if type(inputIn) == list:
      currentInput = inputIn[-1]
    else:
      currentInput = inputIn

    if currentInput.type == 'HistorySet' and self.PreProcessor is None and self.metric is None:
      return self.__runTemporalSciKitLearn(Input)
    else:
      return self.__runSciKitLearn(Input)

  def userInteraction(self):
    """
      A placeholder for allowing user's to interact and tweak the model in-situ
      before saving the analysis results
      @ In, None
      @ Out, None
    """
    pass

  def __runSciKitLearn(self, Input):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject.  This is for SciKitLearn
      @ In, Input, dict, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    self.unSupervisedEngine.features = Input['Features']
    if not self.unSupervisedEngine.amITrained:
      self.unSupervisedEngine.train(Input['Features'], self.metric)
    self.unSupervisedEngine.confidence()

    self.userInteraction()

    outputDict = self.unSupervisedEngine.outputDict

    if 'bicluster' == self.unSupervisedEngine.getDataMiningType():
      self.raiseAnError(RuntimeError, 'Bicluster has not yet been implemented.')

    ## Rename the algorithm output to point to the user-defined label feature
    if 'labels' in outputDict['outputs']:
      outputDict['outputs'][self.labelFeature] = outputDict['outputs'].pop('labels')
    elif 'embeddingVectors' in outputDict['outputs']:
      transformedData = outputDict['outputs'].pop('embeddingVectors')
      reducedDimensionality = transformedData.shape[1]

      for i in range(reducedDimensionality):
        newColumnName = self.labelFeature + str(i + 1)
        outputDict['outputs'][newColumnName] =  transformedData[:, i]

    if self.solutionExport is not None:
      if 'cluster' == self.unSupervisedEngine.getDataMiningType():
        solutionExportDict = self.unSupervisedEngine.metaDict
        if 'clusterCenters' in solutionExportDict:
          centers = solutionExportDict['clusterCenters']

          ## Does skl not provide a correlation between label ids and cluster centers?
          if 'clusterCentersIndices' in solutionExportDict:
            indices = solutionExportDict['clusterCentersIndices']
          else:
            indices = list(range(len(centers)))

          if self.PreProcessor is None:
            for index,center in zip(indices,centers):
              self.solutionExport.updateInputValue(self.labelFeature,index)
              ## Can I be sure of the order of dimensions in the features dict, is
              ## the same order as the data held in the UnSupervisedLearning object?
              for key,value in zip(self.unSupervisedEngine.features.keys(),center):
                self.solutionExport.updateOutputValue(key,value)
          else:
            # if a pre-processor is used it is here assumed that the pre-processor has internally a
            # method (called "inverse") which converts the cluster centers back to their original format. If this method
            # does not exist a warning will be generated
            tempDict = {}
            for index,center in zip(indices,centers):
              tempDict[index] = center
            centers = self.PreProcessor.interface._inverse(tempDict)

            for index,center in zip(indices,centers):
              self.solutionExport.updateInputValue(self.labelFeature,index)

            if self.solutionExport.type == 'HistorySet':
              for hist in centers.keys():
                for key in centers[hist].keys():
                  self.solutionExport.updateOutputValue(key,centers[hist][key])
            else:
              for key in centers.keys():
                if key in self.solutionExport.getParaKeys('outputs'):
                  for value in centers[key]:
                    self.solutionExport.updateOutputValue(key,value)
      elif 'mixture' == self.unSupervisedEngine.getDataMiningType():
        solutionExportDict = self.unSupervisedEngine.metaDict
        mixtureMeans = solutionExportDict['means']
        mixtureCovars = solutionExportDict['covars']
        ## TODO: Export Gaussian centers to SolutionExport
        ## Get the centroids and push them to a SolutionExport data object, if
        ## we have both, also if we have the centers, assume we have the indices
        ## to match them.

        ## Does skl not provide a correlation between label ids and Gaussian
        ## centers?
        indices = list(range(len(mixtureMeans)))
        for index,center in zip(indices,mixtureMeans):
          self.solutionExport.updateInputValue(self.labelFeature,index)
          ## Can I be sure of the order of dimensions in the features dict, is
          ## the same order as the data held in the UnSupervisedLearning
          ## object?
          for key,value in zip(self.unSupervisedEngine.features.keys(),center):
            self.solutionExport.updateOutputValue(key,value)
          ## You may also want to output the covariances of each pair of
          ## dimensions as well
          for i,row in enumerate(self.unSupervisedEngine.features.keys()):
            for joffset,col in enumerate(self.unSupervisedEngine.features.keys()[i:]):
              j = i+joffset
              self.solutionExport.updateOutputValue('cov_'+str(row)+'_'+str(col),mixtureCovars[index][i,j])
      elif 'decomposition' == self.unSupervisedEngine.getDataMiningType():
        solutionExportDict = self.unSupervisedEngine.metaDict

        ## Get the transformation matrix and push it to a SolutionExport
        ## data object.
        ## Can I be sure of the order of dimensions in the features dict, is
        ## the same order as the data held in the UnSupervisedLearning object?
        if 'components' in solutionExportDict:
          components = solutionExportDict['components']
          for row,values in enumerate(components):
            self.solutionExport.updateInputValue(self.labelFeature, row+1)
            for col,value in zip(self.unSupervisedEngine.features.keys(),values):
              self.solutionExport.updateOutputValue(col,value)

            if 'explainedVarianceRatio' in solutionExportDict:
              self.solutionExport.updateOutputValue('ExplainedVarianceRatio',solutionExportDict['explainedVarianceRatio'][row])
    return outputDict

  def __runTemporalSciKitLearn(self, Input):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject.  This is for temporalSciKitLearn
      @ In, Input, dict, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    self.unSupervisedEngine.features = Input['Features']
    self.unSupervisedEngine.pivotVariable = self.pivotVariable

    if not self.unSupervisedEngine.amITrained:
      self.unSupervisedEngine.train(Input['Features'])
    self.unSupervisedEngine.confidence()

    self.userInteraction()

    outputDict = self.unSupervisedEngine.outputDict

    numberOfHistoryStep = self.unSupervisedEngine.numberOfHistoryStep
    numberOfSample = self.unSupervisedEngine.numberOfSample

    if 'bicluster' == self.unSupervisedEngine.getDataMiningType():
      self.raiseAnError(RuntimeError, 'Bicluster has not yet been implemented.')

    ## Rename the algorithm output to point to the user-defined label feature
    # if 'labels' in outputDict:
    #   outputDict['outputs'][self.labelFeature] = outputDict['outputs'].pop('labels')
    # elif 'embeddingVectors' in outputDict['outputs']:
    #   transformedData = outputDict['outputs'].pop('embeddingVectors')
    #   reducedDimensionality = transformedData.shape[1]

    #   for i in range(reducedDimensionality):
    #     newColumnName = self.labelFeature + str(i + 1)
    #     outputDict['outputs'][newColumnName] =  transformedData[:, i]

    if 'labels' in self.unSupervisedEngine.outputDict['outputs'].keys():
      labels = np.zeros(shape=(numberOfSample,numberOfHistoryStep))
      for t in range(numberOfHistoryStep):
        labels[:,t] = self.unSupervisedEngine.outputDict['outputs']['labels'][t]
      outputDict['outputs'][self.labelFeature] = labels
    elif 'embeddingVectors' in outputDict['outputs']:
      transformedData = outputDict['outputs'].pop('embeddingVectors')
      reducedDimensionality = transformedData.values()[0].shape[1]

      for i in range(reducedDimensionality):
        dimensionI = np.zeros(shape=(numberOfSample,numberOfHistoryStep))
        newColumnName = self.labelFeature + str(i + 1)

        for t in range(numberOfHistoryStep):
          dimensionI[:, t] =  transformedData[t][:, i]
        outputDict['outputs'][newColumnName] = dimensionI

    if 'cluster' == self.unSupervisedEngine.getDataMiningType():
      ## SKL will always enumerate cluster centers starting from zero, if this
      ## is violated, then the indexing below will break.
      if 'clusterCentersIndices' in self.unSupervisedEngine.metaDict.keys():
        clusterCentersIndices = self.unSupervisedEngine.metaDict['clusterCentersIndices']

      if 'clusterCenters' in self.unSupervisedEngine.metaDict.keys():
        clusterCenters = self.unSupervisedEngine.metaDict['clusterCenters']
        # Output cluster centroid to solutionExport
        if self.solutionExport is not None:
          ## We will process each cluster in turn
          for clusterIdx in xrange(int(np.max(labels))+1):
            ## First store the label as the input for this cluster
            self.solutionExport.updateInputValue(self.labelFeature,clusterIdx)

            ## The time series will be the first output
            ## TODO: Ensure user requests this
            self.solutionExport.updateOutputValue(self.pivotParameter, self.pivotVariable)

            ## Now we will process each feature available
            ## TODO: Ensure user requests each of these
            for featureIdx, feat in enumerate(self.unSupervisedEngine.features):
              ## We will go through the time series and find every instance
              ## where this cluster exists, if it does not, then we put a NaN
              ## to signal that the information is missing for this timestep
              timeSeries = np.zeros(numberOfHistoryStep)

              for timeIdx in range(numberOfHistoryStep):
                ## Here we use the assumption that SKL provides clusters that
                ## are integer values beginning at zero, which make for nice
                ## indexes with no need to add another layer of obfuscation
                if clusterIdx in clusterCentersIndices[timeIdx]:
                  loc = clusterCentersIndices[timeIdx].index(clusterIdx)
                  timeSeries[timeIdx] = self.unSupervisedEngine.metaDict['clusterCenters'][timeIdx][loc,featureIdx]
                else:
                  timeSeries[timeIdx] = np.nan

              ## In summary, for each feature, we fill a temporary array and
              ## stuff it into the solutionExport, one question is how do we
              ## tell it which item we are exporting? I am assuming that if
              ## I add an input, then I need to do the corresponding
              ## updateOutputValue to associate everything with it? Once I
              ## call updateInputValue again, it will move the pointer? This
              ## needs verified
              self.solutionExport.updateOutputValue(feat, timeSeries)

      if 'inertia' in self.unSupervisedEngine.outputDict.keys():
        inertia = self.unSupervisedEngine.outputDict['inertia']

    elif 'mixture' == self.unSupervisedEngine.getDataMiningType():
      if 'covars' in self.unSupervisedEngine.metaDict.keys():
        mixtureCovars = self.unSupervisedEngine.metaDict['covars']
      else:
        mixtureCovars = None

      if 'precs' in self.unSupervisedEngine.metaDict.keys():
        mixturePrecs = self.unSupervisedEngine.metaDict['precs']
      else:
        mixturePrecs = None

      if 'componentMeanIndices' in self.unSupervisedEngine.metaDict.keys():
        componentMeanIndices = self.unSupervisedEngine.metaDict['componentMeanIndices']
      else:
        componentMeanIndices = None

      if 'means' in self.unSupervisedEngine.metaDict.keys():
        mixtureMeans = self.unSupervisedEngine.metaDict['means']
      else:
        mixtureMeans = None

      # Output cluster centroid to solutionExport
      if self.solutionExport is not None:
        ## We will process each cluster in turn
        for clusterIdx in xrange(int(np.max(componentMeanIndices.values()))+1):
          ## First store the label as the input for this cluster
          self.solutionExport.updateInputValue(self.labelFeature,clusterIdx)

          ## The time series will be the first output
          ## TODO: Ensure user requests this
          self.solutionExport.updateOutputValue(self.pivotParameter, self.pivotVariable)

          ## Now we will process each feature available
          ## TODO: Ensure user requests each of these
          if mixtureMeans is not None:
            for featureIdx, feat in enumerate(self.unSupervisedEngine.features):
              ## We will go through the time series and find every instance
              ## where this cluster exists, if it does not, then we put a NaN
              ## to signal that the information is missing for this timestep
              timeSeries = np.zeros(numberOfHistoryStep)

              for timeIdx in range(numberOfHistoryStep):
                loc = componentMeanIndices[timeIdx].index(clusterIdx)
                timeSeries[timeIdx] = mixtureMeans[timeIdx][loc,featureIdx]

              ## In summary, for each feature, we fill a temporary array and
              ## stuff it into the solutionExport, one question is how do we
              ## tell it which item we are exporting? I am assuming that if
              ## I add an input, then I need to do the corresponding
              ## updateOutputValue to associate everything with it? Once I
              ## call updateInputValue again, it will move the pointer? This
              ## needs verified
              self.solutionExport.updateOutputValue(feat, timeSeries)

          ## You may also want to output the covariances of each pair of
          ## dimensions as well
          if mixtureCovars is not None:
            for i,row in enumerate(self.unSupervisedEngine.features.keys()):
              for joffset,col in enumerate(self.unSupervisedEngine.features.keys()[i:]):
                j = i+joffset
                timeSeries = np.zeros(numberOfHistoryStep)
                for timeIdx in range(numberOfHistoryStep):
                  loc = componentMeanIndices[timeIdx].index(clusterIdx)
                  timeSeries[timeIdx] = mixtureCovars[timeIdx][loc][i,j]
                self.solutionExport.updateOutputValue('cov_'+str(row)+'_'+str(col),timeSeries)
    elif 'decomposition' == self.unSupervisedEngine.getDataMiningType():
      if self.solutionExport is not None:
        solutionExportDict = self.unSupervisedEngine.metaDict
        ## Get the transformation matrix and push it to a SolutionExport
        ## data object.
        ## Can I be sure of the order of dimensions in the features dict, is
        ## the same order as the data held in the UnSupervisedLearning object?
        if 'components' in solutionExportDict:
          components = solutionExportDict['components']

          ## Note, this implies some data exists (Really this information should
          ## be stored in a dictionary to begin with)
          numComponents,numDimensions = components[0].shape

          componentsArray = np.zeros((numberOfHistoryStep,numComponents, numDimensions))
          evrArray = np.zeros((numberOfHistoryStep,numComponents))

          for timeIdx in range(numberOfHistoryStep):
            for componentIdx,values in enumerate(components[timeIdx]):
              componentsArray[timeIdx,componentIdx,:] = values
              evrArray[timeIdx,componentIdx] = solutionExportDict['explainedVarianceRatio'][timeIdx][componentIdx]

          for componentIdx in range(numComponents):
            ## First store the dimension name as the input for this component
            self.solutionExport.updateInputValue(self.labelFeature, componentIdx+1)

            ## The time series will be the first output
            ## TODO: Ensure user requests this
            self.solutionExport.updateOutputValue(self.pivotParameter, self.pivotVariable)

            ## Now we will process each feature available
            ## TODO: Ensure user requests each of these
            for dimIdx,dimName in enumerate(self.unSupervisedEngine.features.keys()):
              values = componentsArray[:,componentIdx,dimIdx]
              self.solutionExport.updateOutputValue(dimName,values)

            if 'explainedVarianceRatio' in solutionExportDict:
              self.solutionExport.updateOutputValue('ExplainedVarianceRatio',evrArray[:,componentIdx])

    return outputDict

try:
  import PySide.QtCore as qtc

  class QDataMining(DataMining,qtc.QObject):
    """
      DataMining class - Computes a hierarchical clustering from an input point
      cloud consisting of an arbitrary number of input parameters
    """
    requestUI = qtc.Signal(str,str,dict)
    def __init__(self, messageHandler):
      """
       Constructor
       @ In, messageHandler, message handler object
       @ Out, None
      """
      DataMining.__init__(self, messageHandler)
      qtc.QObject.__init__(self)
      self.interactive = False

    def _localReadMoreXML(self, xmlNode):
      """
        Function to grab the names of the methods this post-processor will be
        using
        @ In, xmlNode    : Xml element node
        @ Out, None
      """
      DataMining._localReadMoreXML(self, xmlNode)
      for child in xmlNode:
        for grandchild in child:
          if grandchild.tag == 'interactive':
            self.interactive = True

    def _localWhatDoINeed(self):
      """
        This method is a local mirror of the general whatDoINeed method.
        It is implemented by the samplers that need to request special objects
        @ In , None, None
        @ Out, needDict, list of objects needed
      """
      needDict = DataMining._localWhatDoINeed(self)
      needDict['internal'].append((None,'app'))
      return needDict

    def _localGenerateAssembler(self,initDict):
      """
        Generates the assembler.
        @ In, initDict, dict of init objects
        @ Out, None
      """
      DataMining._localGenerateAssembler(self, initDict)
      self.app = initDict['internal']['app']
      if self.app is None:
        self.interactive = False

    def userInteraction(self):
      """
        Launches an interface allowing the user to tweak specific model
        parameters before saving the results to the output object(s).
        @ In, None
        @ Out, None
      """

      ## If it has not been requested, then we are not waiting for a UI,
      ## otherwise the UI has been requested, and we are going to need to wait
      ## for it.
      self.uiDone = not self.interactive

      if self.interactive:

        ## Connect our own signal to the slot on the main thread
        self.requestUI.connect(self.app.createUI)

        ## Connect our own slot to listen for whenver the main thread signals a
        ## window has been closed
        self.app.windowClosed.connect(self.signalDone)

        ## Give this UI a unique id in case other threads are requesting UI
        ##  elements
        uiID = unicode(id(self))

        ## Send the request for a UI thread to the main application
        self.requestUI.emit('HierarchyWindow', uiID,
                            {'views': ['DendrogramView','ScatterView'],
                             'debug': False,
                             'engine': self.unSupervisedEngine})

        ## Spinlock will wait until this instance's window has been closed
        while(not self.uiDone):
          time.sleep(1)

        ## First check that the requested UI exists, and then if that UI has the
        ## requested information, if not proceed as if it were not an
        ## interactive session.
        if uiID in self.app.UIs and hasattr(self.app.UIs[uiID],'level') and self.app.UIs[uiID].level is not None:
          self.initializationOptionDict['KDD']['level'] = self.app.UIs[uiID].level

    def signalDone(self,uiID):
      """
        In Qt language, this is a slot that will accept a signal from the UI
        saying that it has completed, thus allowing the computation to begin
        again with information updated by the user in the UI.
        @In, uiID, string, the ID of the user interface that signaled its
            completion. Thus, if several UI windows are open, we don't proceed,
            until the correct one has signaled it is done.
        @Out, None
      """
      if uiID == unicode(id(self)):
        self.uiDone = True
except ImportError as e:
  pass

class RavenOutput(BasePostProcessor):
  """
    This postprocessor collects the outputs of RAVEN runs (XML format) and turns entries into a PointSet
    Someday maybe it should do history sets too.
  """
  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    BasePostProcessor.__init__(self, messageHandler)
    self.printTag = 'POSTPROCESSOR RAVENOUTPUT'
    self.IDType = 'int'
    self.files = {}
      # keyed by ID, which gets you to... (self.files[ID])
      #   name: RAVEN name for file (from input)
      #   fileObject: FileObject
      #   paths: {varName:'path|through|xml|to|var'}
    self.dynamic = False #if true, reading in pivot as input and values as outputs

  def initialize(self,runInfo,inputs,initDict):
    """
      Method to initialize pp
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    BasePostProcessor.initialize(self, runInfo, inputs, initDict)
    #assign File objects to their proper place
    for id,fileDict in self.files.items():
      found = False
      for i,input in enumerate(inputs):
        #skip things that aren't files
        if not isinstance(input,Files.File):
          continue
        #assign pointer to file object if match found
        if input.name == fileDict['name']:
          self.files[id]['fileObject'] = input
          found = True
          break
      if not found:
        self.raiseAnError(IOError,'Did not find file named "%s" among the Step inputs!' % (input.name))

  def _localReadMoreXML(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    #check if in dynamic mode; default is False
    dynamicNode = xmlNode.find('dynamic')
    if dynamicNode is not None:
      #could specify as true/false or just have the node present
      text = dynamicNode.text
      if text is not None:
        if text not in utils.stringsThatMeanFalse():
          self.dynamic = True
      else:
        self.dynamic = True
    numberOfSources = 0
    for child in xmlNode:
      #if dynamic, accept a single file as <File ID="1" name="myOut.xml">
      #if not dynamic, accept a list of files
      if child.tag == 'File':
        numberOfSources += 1
        if 'name' not in child.attrib.keys():
          self.raiseAnError(IOError,'Each "File" must have an associated "name"; missing for',child.tag,child.text)
        #make sure you provide an ID and a file name
        if 'ID' not in child.attrib.keys():
          id = 0
          while id in self.files.keys():
            id += 1
          self.raiseAWarning(IOError,'Each "File" entry must have an associated "ID"; missing for',child.tag,child.attrib['name'],'so ID is set to',id)
        else:
          #assure ID is a number, since it's going into a data object
          id = child.attrib['ID']
          try:
            id = float(id)
          except ValueError:
            self.raiseAnError(IOError,'ID for "'+child.text+'" is not a valid number:',id)
          #if already used, raise an error
          if id in self.files.keys():
            self.raiseAnError(IOError,'Multiple File nodes have the same ID:',child.attrib('ID'))
        #store id,filename pair
        self.files[id] = {'name':child.attrib['name'].strip(), 'fileObject':None, 'paths':{}}
        #user provides loading information as <output name="variablename">ans|pearson|x</output>
        for cchild in child:
          if cchild.tag == 'output':
            #make sure you provide a label for this data array
            if 'name' not in cchild.attrib.keys():
              self.raiseAnError(IOError,'Must specify a "name" for each "output" block!  Missing for:',cchild.text)
            varName = cchild.attrib['name'].strip()
            if varName in self.files[id]['paths'].keys():
              self.raiseAnError(IOError,'Multiple "output" blocks for "%s" have the same "name":' %self.files[id]['name'],varName)
            self.files[id]['paths'][varName] = cchild.text.strip()
    #if dynamic, only one File can be specified currently; to fix this, how do you handle different-lengthed times in same data object?
    if self.dynamic and numberOfSources > 1:
      self.raiseAnError(IOError,'For Dynamic reading, only one "File" node can be specified!  Got',numberOfSources,'nodes.')
    # check there are entries for each
    if len(self.files)<1:
      self.raiseAWarning('No files were specified to read from!  Nothing will be done...')
    # if no outputs listed, remove file from list and warn
    toRemove=[]
    for id,fileDict in self.files.items():
      if len(fileDict['paths'])<1:
        self.raiseAWarning('No outputs were specified for File with ID "%s"!  No extraction will be performed for this file...' %str(id))
        toRemove.append(id)
    for rem in toRemove:
      del self.files[id]

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """
    # outputs are realizations that will got into data object
    outputDict={'realizations':[]}
    if self.dynamic:
      #outputs are basically a point set with pivot as input and requested XML path entries as output
      fileName = self.files.values()[0]['fileObject'].getAbsFile()
      root,_ = xmlUtils.loadToTree(fileName)
      #determine the pivot parameter
      pivot = root[0].tag
      numPivotSteps = len(root)
      #read from each iterative pivot step
      for p,pivotStep in enumerate(root):
        realization = {'inputs':{},'outputs':{},'metadata':{'loadedFromRavenFile':fileName}}
        realization['inputs'][pivot] = float(pivotStep.attrib['value'])
        for name,path in self.files.values()[0]['paths'].items():
          desiredNode = self._readPath(pivotStep,path,fileName)
          realization['outputs'][name] = float(desiredNode.text)
        outputDict['realizations'].append(realization)
    else:
      # each ID results in a realization for the requested attributes
      for id,fileDict in self.files.items():
        realization = {'inputs':{'ID':id},'outputs':{},'metadata':{'loadedFromRavenFile':str(fileDict['fileObject'])}}
        for varName,path in fileDict['paths'].items():
          #read the value from the file's XML
          root,_ = xmlUtils.loadToTree(fileDict['fileObject'].getAbsFile())
          desiredNode = self._readPath(root,path,fileDict['fileObject'].getAbsFile())
          realization['outputs'][varName] = float(desiredNode.text)
        outputDict['realizations'].append(realization)
    return outputDict

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    if finishedJob.getEvaluation() == -1: self.raiseAnError(RuntimeError, 'No available Output to collect (Run probably is not finished yet)')
    realizations = finishedJob.getEvaluation()[1]['realizations']
    for real in realizations:
      for key in output.getParaKeys('inputs'):
        if key not in real['inputs'].keys(): self.raiseAnError(RuntimeError, 'Requested input variable '+key+' has not been extracted. Check the consistency of your input')
        output.updateInputValue(key,real['inputs'][key])
      for key in output.getParaKeys('outputs'):
        if key not in real['outputs'].keys(): self.raiseAnError(RuntimeError, 'Requested output variable '+key+' has not been extracted. Check the consistency of your input')
        output.updateOutputValue(key,real['outputs'][key])
      for key,val in real['metadata'].items():
        output.updateMetadata(key,val)

  def _readPath(self,root,inpPath,fileName):
    """
      Reads in values from XML tree.
      @ In, root, xml.etree.ElementTree.Element, node to start from
      @ In, inPath, string, |-separated list defining path from root (not including root)
      @ In, fileName, string, used in error
      @ Out, desiredNode, xml.etree.ElementTree.Element, desired node
    """
    #improve path format
    path = '|'.join(c.strip() for c in inpPath.strip().split('|'))
    desiredNode = xmlUtils.findPath(root,path)
    if desiredNode is None:
      self.raiseAnError(RuntimeError,'Did not find "%s|%s" in file "%s"' %(root.tag,path,fileName))
    return desiredNode





"""
 Interface Dictionary (factory) (private)
"""
__base = 'PostProcessor'
__interFaceDict = {}
__interFaceDict['SafestPoint'              ] = SafestPoint
__interFaceDict['LimitSurfaceIntegral'     ] = LimitSurfaceIntegral
__interFaceDict['BasicStatistics'          ] = BasicStatistics
__interFaceDict['InterfacedPostProcessor'  ] = InterfacedPostProcessor
__interFaceDict['LimitSurface'             ] = LimitSurface
__interFaceDict['ComparisonStatistics'     ] = ComparisonStatistics
__interFaceDict['External'                 ] = ExternalPostProcessor
try:
  __interFaceDict['TopologicalDecomposition' ] = QTopologicalDecomposition
except NameError:
  __interFaceDict['TopologicalDecomposition' ] = TopologicalDecomposition

try:
  __interFaceDict['DataMining'               ] = QDataMining
except NameError:
  __interFaceDict['DataMining'               ] = DataMining

__interFaceDict['ImportanceRank'           ] = ImportanceRank
__interFaceDict['RavenOutput'              ] = RavenOutput
__knownTypes = __interFaceDict.keys()

def knownTypes():
  """
    Return the known types
    @ In, None
    @ Out, __knownTypes, list, list of known types
  """
  return __knownTypes

def returnInstance(Type, caller):
  """
    function used to generate a Sampler class
    @ In, Type, string, Sampler type
    @ Out, returnInstance, instance, Instance of the Specialized Sampler class
  """
  try: return __interFaceDict[Type](caller.messageHandler)
  except KeyError: caller.raiseAnError(NameError, 'not known ' + __base + ' type ' + Type)
