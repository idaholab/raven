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
import xarray
import math
import os
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .PostProcessor import PostProcessor
from .BasicStatistics import BasicStatistics
from utils import InputData
import LearningGate
import Files
import Runners
#Internal Modules End--------------------------------------------------------------------------------


class LimitSurfaceIntegral(PostProcessor):
  """
    This post-processor computes the n-dimensional integral of a Limit Surface
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
    inputSpecification = super(LimitSurfaceIntegral, cls).getInputSpecification()

    LSIVariableInput = InputData.parameterInputFactory("variable")
    LSIVariableInput.addParam("name", InputData.StringType)
    LSIDistributionInput = InputData.parameterInputFactory("distribution", contentType=InputData.StringType)
    LSIDistributionInput.addParam("class", InputData.StringType, True)
    LSIDistributionInput.addParam("type", InputData.StringType, True)
    LSIVariableInput.addSub(LSIDistributionInput)
    LSILowerBoundInput = InputData.parameterInputFactory("lowerBound", contentType=InputData.FloatType)
    LSIVariableInput.addSub(LSILowerBoundInput)
    LSIUpperBoundInput = InputData.parameterInputFactory("upperBound", contentType=InputData.FloatType)
    LSIVariableInput.addSub(LSIUpperBoundInput)
    inputSpecification.addSub(LSIVariableInput)

    LSIToleranceInput = InputData.parameterInputFactory("tolerance", contentType=InputData.FloatType)
    inputSpecification.addSub(LSIToleranceInput)

    LSIIntegralTypeInput = InputData.parameterInputFactory("integralType", contentType=InputData.StringType)
    inputSpecification.addSub(LSIIntegralTypeInput)

    LSISeedInput = InputData.parameterInputFactory("seed", contentType=InputData.IntegerType)
    inputSpecification.addSub(LSISeedInput)

    LSITargetInput = InputData.parameterInputFactory("target", contentType=InputData.StringType)
    inputSpecification.addSub(LSITargetInput)

    LSIOutputNameInput = InputData.parameterInputFactory("outputName", contentType=InputData.StringType)
    inputSpecification.addSub(LSIOutputNameInput)

    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.variableDist = {}  # dictionary created upon the .xml input file reading. It stores the distributions for each variable.
    self.target = None  # target that defines the f(x1,x2,...,xn)
    self.tolerance = 0.0001  # integration tolerance
    self.integralType = 'montecarlo'  # integral type (which alg needs to be used). Either montecarlo or quadrature(quadrature not yet)
    self.seed = 20021986  # seed for montecarlo
    self.matrixDict = {}  # dictionary of arrays and target
    self.lowerUpperDict = {}
    self.functionS = None
    self.computationPrefix = None
    self.stat = BasicStatistics(self.messageHandler)  # instantiation of the 'BasicStatistics' processor, which is used to compute the pb given montecarlo evaluations
    self.stat.what = ['expectedValue']
    self.addAssemblerObject('distribution','-n', newXmlFlg = True)
    self.printTag = 'POSTPROCESSOR INTEGRAL'

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """
    paramInput = LimitSurfaceIntegral.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
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
      elif child.getName() == 'outputName':
        self.computationPrefix = child.value
      else:
        self.raiseAnError(NameError, 'invalid or missing labels after the variables call. Only "variable" is accepted.tag: ' + child.getName())
      # if no distribution, we look for the integration domain in the input
      if varName != None:
        if self.variableDist[varName] == None:
          if 'lowerBound' not in self.lowerUpperDict[varName].keys() or 'upperBound' not in self.lowerUpperDict[varName].keys():
            self.raiseAnError(NameError, 'either a distribution name or lowerBound and upperBound need to be specified for variable ' + varName)
    if self.computationPrefix == None:
      self.raiseAnError(IOError,'The required XML node <outputName> has not been inputted!!!')
    if self.target == None:
      self.raiseAWarning('integral target has not been provided. The postprocessor is going to take the last output it finds in the provided limitsurface!!!')

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
      self.stat.toDo = {'expectedValue':[{'targets':set([self.target]), 'prefix':self.computationPrefix}]}
      self.stat.initialize(runInfo, inputs, initDict)
    self.functionS = LearningGate.returnInstance('SupervisedGate','SciKitLearn', self, **{'SKLtype':'neighbors|KNeighborsClassifier', 'Features':','.join(list(self.variableDist.keys())), 'Target':self.target})
    self.functionS.train(self.matrixDict)
    self.raiseADebug('DATA SET MATRIX:')
    self.raiseADebug(self.matrixDict)
    for varName, distName in self.variableDist.items():
      if distName != None:
        self.variableDist[varName] = self.retrieveObjectFromAssemblerDict('distribution', distName)
        self.lowerUpperDict[varName]['lowerBound'] = self.variableDist[varName].lowerBound
        self.lowerUpperDict[varName]['upperBound'] = self.variableDist[varName].upperBound

  def inputToInternal(self, currentInput):
    """
     Method to convert an input object into the internal format that is
     understandable by this pp.
     The resulting converted object is stored as an attribute of this class
     @ In, currentInput, object, an object that needs to be converted
     @ Out, None
    """
    if len(currentInput) > 1:
      self.raiseAnError(IOError,"This PostProcessor can accept only a single input! Got: "+ str(len(currentInput))+"!")
    item = currentInput[0]
    if item.type == 'PointSet':
      if not set(item.getVars('input')) == set(self.variableDist.keys()):
        self.raiseAnError(IOError, 'The variables inputted and the features in the input PointSet ' + item.name + 'do not match!!!')
      outputKeys = item.getVars('output')
      if self.target is None:
        self.target = utils.first(outputKeys)
      elif self.target not in outputKeys:
        self.raiseAnError(IOError, 'The target ' + self.target + 'is not present among the outputs of the PointSet ' + item.name)
      # construct matrix
      dataSet = item.asDataset()
      self.matrixDict = {varName: dataSet[varName].values for varName in self.variableDist}
      responseArray = dataSet[self.target].values
      if len(np.unique(responseArray)) != 2:
        self.raiseAnError(IOError, 'The target ' + self.target + ' needs to be a classifier output (-1 +1 or 0 +1)!')
      responseArray[responseArray == -1] = 0.0
      self.matrixDict[self.target] = responseArray
    else:
      self.raiseAnError(IOError, 'Only PointSet is accepted as input!!!!')

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
        if self.variableDist[varName] == None:
          randomMatrix[:, index] = randomMatrix[:, index] * (self.lowerUpperDict[varName]['upperBound'] - self.lowerUpperDict[varName]['lowerBound']) + self.lowerUpperDict[varName]['lowerBound']
        else:
          f = np.vectorize(self.variableDist[varName].ppf, otypes=[np.float])
          randomMatrix[:, index] = f(randomMatrix[:, index])
        tempDict[varName] = randomMatrix[:, index]
      pb = self.stat.run({'targets':{self.target:xarray.DataArray(self.functionS.evaluate(tempDict)[self.target])}})[self.computationPrefix +"_"+self.target]
    else:
      self.raiseAnError(NotImplemented, "quadrature not yet implemented")
    return pb

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "No available output to collect (run possibly not finished yet)")
    pb = evaluation[1]
    lms = evaluation[0][0]
    if output.type == 'PointSet':
      # we store back the limitsurface
      dataSet = lms.asDataset()
      loadDict = {key: dataSet[key].values for key in lms.getVars()}
      loadDict[self.computationPrefix] = np.full(len(lms), pb)
      output.load(loadDict,'dict')
    # NB I keep this commented part in case we want to keep the possibility to have outputfiles for PP
    #elif isinstance(output,Files.File):
    #  headers = lms.getParaKeys('inputs') + lms.getParaKeys('outputs')
    #  if 'EventProbability' not in headers:
    #    headers += ['EventProbability']
    #  stack = [None] * len(headers)
    #  output.close()
    #  # If the file already exist, we will erase it.
    #  if os.path.exists(output.getAbsFile()):
    #    self.raiseAWarning('File %s already exists, this file will be erased!' %output.getAbsFile())
    #    output.open('w')
    #    output.close()
    #  outIndex = 0
    #  for key, value in lms.getParametersValues('input').items():
    #    stack[headers.index(key)] = np.asarray(value).flatten()
    #  for key, value in lms.getParametersValues('output').items():
    #    stack[headers.index(key)] = np.asarray(value).flatten()
    #    outIndex = headers.index(key)
    #  stack[headers.index('EventProbability')] = np.array([pb] * len(stack[outIndex])).flatten()
    #  stacked = np.column_stack(stack)
    #  np.savetxt(output, stacked, delimiter = ',', header = ','.join(headers),comments='')
    #  #N.B. without comments='' you get a "# " at the top of the header row
    else:
      self.raiseAnError(Exception, self.type + ' accepts PointSet only')
