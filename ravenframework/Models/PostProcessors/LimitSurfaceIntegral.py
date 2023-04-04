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
import numpy as np
import xarray
import math

from .PostProcessorInterface import PostProcessorInterface
from ...utils import InputData, InputTypes
from ...SupervisedLearning import factory as romFactory


class LimitSurfaceIntegral(PostProcessorInterface):
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
    inputSpecification = super().getInputSpecification()

    LSIVariableInput = InputData.parameterInputFactory("variable")
    LSIVariableInput.addParam("name", InputTypes.StringType)
    LSIDistributionInput = InputData.parameterInputFactory("distribution", contentType=InputTypes.StringType)
    LSIDistributionInput.addParam("class", InputTypes.StringType, True)
    LSIDistributionInput.addParam("type", InputTypes.StringType, True)
    LSIVariableInput.addSub(LSIDistributionInput)
    LSILowerBoundInput = InputData.parameterInputFactory("lowerBound", contentType=InputTypes.FloatType)
    LSIVariableInput.addSub(LSILowerBoundInput)
    LSIUpperBoundInput = InputData.parameterInputFactory("upperBound", contentType=InputTypes.FloatType)
    LSIVariableInput.addSub(LSIUpperBoundInput)
    inputSpecification.addSub(LSIVariableInput)

    LSIToleranceInput = InputData.parameterInputFactory("tolerance", contentType=InputTypes.FloatType)
    inputSpecification.addSub(LSIToleranceInput)

    LSIIntegralTypeInput = InputData.parameterInputFactory("integralType", contentType=InputTypes.StringType)
    inputSpecification.addSub(LSIIntegralTypeInput)

    LSISeedInput = InputData.parameterInputFactory("seed", contentType=InputTypes.IntegerType)
    inputSpecification.addSub(LSISeedInput)

    LSITargetInput = InputData.parameterInputFactory("target", contentType=InputTypes.StringType)
    inputSpecification.addSub(LSITargetInput)

    LSIOutputNameInput = InputData.parameterInputFactory("outputName", contentType=InputTypes.StringType)
    inputSpecification.addSub(LSIOutputNameInput)

    LSIOutputNameInput = InputData.parameterInputFactory("computeBounds", contentType=InputTypes.BoolType)
    inputSpecification.addSub(LSIOutputNameInput)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    from ...Models.PostProcessors import factory as ppFactory # delay import to allow definition
    self.variableDist = {}  # dictionary created upon the .xml input file reading. It stores the distributions for each variable.
    self.target = None  # target that defines the f(x1,x2,...,xn)
    self.tolerance = 0.0001  # integration tolerance
    self.integralType = 'montecarlo'  # integral type (which alg needs to be used). Either montecarlo or quadrature(quadrature not yet)
    self.seed = 20021986  # seed for montecarlo
    self.matrixDict = {}  # dictionary of arrays and target
    self.computeErrrorBounds = False #  compute the bounding error?
    self.lowerUpperDict = {} # dictionary of lower and upper bounds (used if no distributions are inputted)
    self.functionS = None # evaluation classifier for the integration
    self.errorModel = None # classifier used for the error estimation
    self.computationPrefix = None # output prefix for the storage of the probability and, if requested, bounding error
    self.stat = ppFactory.returnInstance('BasicStatistics')  # instantiation of the 'BasicStatistics' processor, which is used to compute the pb given montecarlo evaluations
    self.stat.what = ['expectedValue'] # expected value calculation
    self.addAssemblerObject('distribution', InputData.Quantity.zero_to_infinity) # distributions are optional
    self.printTag = 'POSTPROCESSOR INTEGRAL' # print tag

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
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
      elif child.getName() == 'computeBounds':
        self.computeErrrorBounds = child.value
      else:
        self.raiseAnError(NameError, 'invalid or missing labels after the variables call. Only "variable" is accepted.tag: ' + child.getName())
      # if no distribution, we look for the integration domain in the input
      if varName != None:
        if self.variableDist[varName] == None:
          if 'lowerBound' not in self.lowerUpperDict[varName].keys() or 'upperBound' not in self.lowerUpperDict[varName].keys():
            self.raiseAnError(NameError, 'either a distribution name or lowerBound and upperBound need to be specified for variable ' + varName)
    if self.computationPrefix is None:
      self.raiseAnError(IOError,'The required XML node <outputName> has not been inputted!!!')
    if self.target is None:
      self.raiseAWarning('integral target has not been provided. The postprocessor is going to take the last output it finds in the provided limitsurface!!!')
    True

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
    self.functionS = romFactory.returnInstance('KNeighborsClassifier')
    paramDict = {'Features':list(self.variableDist.keys()), 'Target':[self.target]}
    self.functionS.initializeFromDict(paramDict)
    ##FIXME set n_jobs = -1 will cause "ValueError: unsupported pickle protocol: 5"
    # settings = {'n_jobs': -1}
    settings = {}
    self.functionS.initializeModel(settings)
    self.functionS.train(self.matrixDict)
    self.raiseADebug('DATA SET MATRIX:')
    self.raiseADebug(self.matrixDict)
    if self.computeErrrorBounds:
      #  create a model for computing the "error"
      self.errorModel = romFactory.returnInstance('KNeighborsClassifier')
      paramDict = {'Features':list(self.variableDist.keys()), 'Target':[self.target]}
      self.errorModel.initializeFromDict(paramDict)
      ##FIXME set n_jobs = -1 will cause "ValueError: unsupported pickle protocol: 5"
      # settings = {'weights': 'distance', 'n_jobs': -1}
      settings = {'weights': 'distance'}
      self.errorModel.initializeModel(settings)
      #modify the self.matrixDict to compute half of the "error"
      indecesToModifyOnes = np.argwhere(self.matrixDict[self.target] > 0.).flatten()
      res = np.concatenate((np.ones(len(indecesToModifyOnes)), np.zeros(len(indecesToModifyOnes))))
      modifiedMatrixDict = {}
      for key in self.matrixDict:
        avg = np.average(self.matrixDict[key][indecesToModifyOnes])
        modifiedMatrixDict[key] = np.concatenate((self.matrixDict[key][indecesToModifyOnes], self.matrixDict[key][indecesToModifyOnes]
                                                  + (self.matrixDict[key][indecesToModifyOnes]/avg * 1.e-14))) if key != self.target else res
      self.errorModel.train(modifiedMatrixDict)

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
      @ Out, boundError, float, optional, error bound (maximum error of the computed probability)
    """
    pb, boundError = None, None
    if self.integralType == 'montecarlo':
      tempDict = {}
      randomMatrix = np.random.rand(int(math.ceil(1.0 / self.tolerance**2)), len(self.variableDist.keys()))
      for index, varName in enumerate(self.variableDist.keys()):
        if self.variableDist[varName] == None:
          randomMatrix[:, index] = randomMatrix[:, index] * (self.lowerUpperDict[varName]['upperBound'] - self.lowerUpperDict[varName]['lowerBound']) + self.lowerUpperDict[varName]['lowerBound']
        else:
          f = np.vectorize(self.variableDist[varName].ppf, otypes=[np.float64])
          randomMatrix[:, index] = f(randomMatrix[:, index])
        tempDict[varName] = randomMatrix[:, index]
      pb = self.stat.run({'targets':{self.target:xarray.DataArray(self.functionS.evaluate(tempDict)[self.target])}})[self.computationPrefix +"_"+self.target]
      if self.errorModel:
        boundError = abs(pb-self.stat.run({'targets':{self.target:xarray.DataArray(self.errorModel.evaluate(tempDict)[self.target])}})[self.computationPrefix +"_"+self.target])
    else:
      self.raiseAnError(NotImplemented, "quadrature not yet implemented")
    return pb, boundError

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    pb, boundError = evaluation[1]
    lms = evaluation[0][0]
    if output.type == 'PointSet':
      # we store back the limitsurface
      dataSet = lms.asDataset()
      loadDict = {key: dataSet[key].values for key in lms.getVars()}
      loadDict[self.computationPrefix] = np.full(len(lms), pb)
      if self.computeErrrorBounds:
        if self.computationPrefix+"_err" not in output.getVars():
          self.raiseAWarning('ERROR Bounds have been computed but the output DataObject does not request the variable: "', self.computationPrefix+"_err", '"!')
        else:
          loadDict[self.computationPrefix+"_err"] = np.full(len(lms), boundError)
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
