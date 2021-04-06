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
  Created on March 20, 2021
  @author: alfoa
  description: Postprocessor named Validation. This postprocessor is aimed to
               to represent a gate for any validation tecniques and processes
"""
                                                                                
#External Modules---------------------------------------------------------------
import numpy as np
import copy
import time
import xarray as xr
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
import validationAlgorithms
#from .validationAlgorithms import
#from .validationAlgorithms import DSS
#from .validationAlgorithms import DSS
#from .validationAlgorithms import PCM
#from .validationAlgorithms import Representativity
from utils import utils, mathUtils
from utils import InputData, InputTypes
import MetricDistributor
#Internal Modules End-----------------------------------------------------------

class ValidationGate(PostProcessor):
  """
    Validation class. It will apply the specified validation algorithms in
    the models to a dataset, each specified algorithm's output can be loaded to
    dataObject.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, specs, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    ## This will replace the lines above
    specs = super(Validation, cls).getInputSpecification()
    preProcessorInput = InputData.parameterInputFactory("PreProcessor", contentType=InputTypes.StringType)
    preProcessorInput.addParam("class", InputTypes.StringType)
    preProcessorInput.addParam("type", InputTypes.StringType)
    pivotParameterInput = InputData.parameterInputFactory("pivotParameter", contentType=InputTypes.StringType)
    specs.addSub(pivotParameterInput)
    metricInput = InputData.parameterInputFactory("Metric", contentType=InputTypes.StringType)
    metricInput.addParam("class", InputTypes.StringType)
    metricInput.addParam("type", InputTypes.StringType)
    specs.addSub(metricInput)
    # registration of validation algorithm
    for typ in validationAlgorithms.factory.knownTypes():
      algoInput = validationAlgorithms.factory.returnClass(typ)
    specs.addSub(algoInput.getInputSpecification())
    
    
    for algo in _factoryTypes()
      algoInput = InputData.parameterInputFactory(algo)
      algoInput.addParam(_returnClass(algo).getInputSpecification())
      specs.addParam("ValidationAlgorithm",algoInput)
    specs.addSub(kddInput)
    preProcessorInput = InputData.parameterInputFactory("PreProcessor", contentType=InputTypes.StringType)
    preProcessorInput.addParam("class", InputTypes.StringType)
    preProcessorInput.addParam("type", InputTypes.StringType)

    specs.addSub(preProcessorInput)

    return specs

  def __init__(self, runInfoDict):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, runInfoDict)
    self.printTag = 'POSTPROCESSOR VALIDATION'

    #self.addAssemblerObject('PreProcessor', InputData.Quantity.zero_to_one)
    self.addAssemblerObject('Metric', InputData.Quantity.zero_to_one)

    self.solutionExport = None  ## A data object to hold derived info about the algorithm being performed,
                                ## e.g., cluster centers or a projection matrix for dimensionality reduction methods

    self.PreProcessor = None    ## Instance of PreProcessor, default is None
    self.metric = None          ## Instance of Metric, default is None
    self.pivotParameter = None  ## default pivotParameter for HistorySet
    self._type = None           ## the type of library that are used for validation, i.e. DSS

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

  def inputToInternalForHistorySet(self,currentInput):
    """
      Function to convert the input history set into a format that this
      post-processor can understand
      @ In, currentInput, object, DataObject of currentInput
      @ Out, inputDict, dict, an input dictionary that this post-processor can process
    """
    dataSet = currentInput.asDataset()

    return inputDict

  def inputToInternalForPointSet(self,currentInput):
    """
      Function to convert the input point set into a format that this
      post-processor can understand
      @ In, currentInput, object, DataObject of currentInput
      @ Out, inputDict, dict, an input dictionary that this post-processor can process
    """
    ## Get what is available in the data object being operated on
    ## This is potentially more information than we need at the moment, but
    ## it will make the code below easier to read and highlights where objects
    ## are reused more readily

    data = currentInput.asDataset()

  def inputToInternalForPreProcessor(self,currentInput):
    """
      Function to convert the received input into a format that this
      post-processor can understand
      @ In, currentInput, object, DataObject of currentInput
      @ Out, inputDict, dict, an input dictionary that this post-processor can process
    """
    inputDict = {'Features': {}, 'parameters': {}, 'Labels': {}, 'metadata': {}}

    return inputDict

  def inputToInternal(self, currentInp):
    """
      Function to convert the received input into a format this object can
      understand
      @ In, currentInp, list or DataObjects, Some form of data object or list of
        data objects handed to the post-processor
      @ Out, inputDict, dict, An input dictionary this object can process
    """

    if type(currentInp) == list:
      if len(currentInp) > 1:
        self.raiseAnError(IOError, "Only one input is allowed for this post-processor: ", self.name)
      currentInput = currentInp[-1]
    else:
      currentInput = currentInp

    if hasattr(currentInput, 'type'):
      if currentInput.type == 'HistorySet':
        return self.inputToInternalForHistorySet(currentInput)

      elif currentInput.type == 'PointSet':
        return self.inputToInternalForPointSet(currentInput)

    elif type(currentInp) == dict:
      if 'Features' in currentInput.keys():
        return currentInput

    elif isinstance(currentInp, Files.File):
      self.raiseAnError(IOError, 'Validation PP: this PP does not support files as input.')

    elif currentInput.type == 'HDF5':
      self.raiseAnError(IOError, 'Validation PP: this PP does not support HDF5 Objects as input.')

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the DataMining pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)
    if "SolutionExport" in initDict:
      self.solutionExport = initDict["SolutionExport"]
    if "PreProcessor" in self.assemblerDict:
      self.PreProcessor = self.assemblerDict['PreProcessor'][0][3]
    if 'Metric' in self.assemblerDict:
      self.metric = self.assemblerDict['Metric'][0][3]

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    PostProcessor._handleInput(self, paramInput)

    self.initializationOptionDict = {}
    for child in paramInput.subparts:
      if child.getName() == 'ValidationAlgorithm':
        if len(child.parameterValues) > 0:
          ### inquire algorithm and pass metric and pre-processor if any
          ###
          ###
          ###
      elif child.getName() == 'pivotParameter':
        self.pivotParameter = child.value
    if not hasattr(self, 'pivotParameter'):
      #TODO, if doing time dependent data mining that needs this, an error
      # should be thrown
      self.pivotParameter = None
    if self._type:
      self.algoValidation = _returnInstance(self._type,**self.initializationOptionDict['ValidationAlgorithm'])
    else:
      self.raiseAnError(IOError, 'No Validation Algorithm is supplied!')

  def collectOutput(self, finishedJob, outputObject):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler
        object that is in charge of running this post-processor
      @ InOut, outputObject, dataObjects, A reference to an object where we want
        to place our computed results
    """
    if len(outputObject) !=0:
      self.raiseAnError(IOError,"There is some information already stored in the DataObject",outputObject.name, \
              "the calculations from PostProcessor",self.name, " can not be stored in this DataObject!", \
              "Please provide a new empty DataObject for this PostProcessor!")
    ## When does this actually happen?
    evaluation = finishedJob.getEvaluation()
    inputObject, validationDict = evaluation

    if inputObject.type != outputObject.type:
      self.raiseAnError(IOError,"The type of output DataObject",outputObject.name,"is not consistent with input",\
              "DataObject type, i.e. ",outputObject.type,"!=",inputObject.type)
    rlzs = {}
    # first create a new dataset from copying input data object
    dataset = inputObject.asDataset().copy(deep=True)
    sampleTag = inputObject.sampleTag
    sampleCoord = dataset[sampleTag].values
    availVars = dataset.data_vars.keys()
    # update variable values if the values in the dataset are different from the values in the dataMineDict
    # dataMineDict stores all the information generated by the datamining algorithm
    if outputObject.type == 'PointSet':
      for key,value in dataMineDict['outputs'].items():
        if key in availVars and not np.array_equal(value,dataset[key].values):
          newDA = xr.DataArray(value,dims=(sampleTag),coords={sampleTag:sampleCoord})
          dataset = dataset.drop(key)
          dataset[key] = newDA
        elif key not in availVars:
          newDA = xr.DataArray(value,dims=(sampleTag),coords={sampleTag:sampleCoord})
          dataset[key] = newDA
    elif outputObject.type == 'HistorySet':
      for key,values in dataMineDict['outputs'].items():
        if key not in availVars:
          expDict = {}
          for index, value in enumerate(values):
            timeLength = len(self.pivotVariable[index])
            arrayBase = value * np.ones(timeLength)
            xrArray = xr.DataArray(arrayBase,dims=(self.pivotParameter), coords=[self.pivotVariable[index]])
            expDict[sampleCoord[index]] = xrArray
          ds = xr.Dataset(data_vars=expDict)
          ds = ds.to_array().rename({'variable':sampleTag})
          dataset[key] = ds
    else:
      self.raiseAnError(IOError, 'Unrecognized type for output data object ', outputObject.name, \
              '! Available type are HistorySet or PointSet!')

    outputObject.load(dataset,style='dataset')

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
    evaluation = self.__runValidation(Input)
    return evaluation


  def __runValidation(self, Input):
    """
      This method executes the postprocessor action. In this case it loads the
      results to specified dataObject.  This is for SciKitLearn
      @ In, Input, dict, dictionary of data to process
      @ Out, outputDict, dict, dictionary containing the post-processed results
    """

    return outputDict




__availableValidationAlgorithms = {}
__availableValidationAlgorithms['DSS'] = DSS
#__availableValidationAlgorithms['PCM'] = PCM
#__availableValidationAlgorithms['Representativity'] = Representativity

def _factoryTypes():
  """
    Factor of validation algorithms
  """
  return list(__availableValidationAlgorithms.keys())

def _returnClass(cls, name):
  """
    Return instance of validation algorithms
  """
  if name not in _factoryValidationAlgorithms():
    cls.raiseAnError("{} validation algorithm not available!".format(name))
  return __availableValidationAlgorithms[name]

def _returnInstance(cls, name, **kwargs):
  """
    Return instance of validation algorithms
  """
  if name not in _factoryValidationAlgorithms():
    cls.raiseAnError("{} validation algorithm not available!".format(name))
  return __availableValidationAlgorithms[name](**kwargs)
