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
Created on Jan 29, 2018

@author: Congjian Wang
"""
import copy
import numpy as np

from utils import InputData, InputTypes, utils
from .PostProcessorInterface import PostProcessorInterface

class DataClassifier(PostProcessorInterface):
  """
    This Post-Processor performs data classification based on given classifier.
    In order to use this interface post-processor, the users need to provide
    two data objects, one (only PointSet is allowed) is used to construct the
    classifier that will be used to label the data in the second data object
    (both PointSet and HistorySet are allowed).
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
    inputSpecification = super().getInputSpecification()
    VariableInput = InputData.parameterInputFactory("variable", contentType=InputTypes.StringType)
    VariableInput.addParam("name", InputTypes.StringType, True)
    FunctionInput = InputData.parameterInputFactory("Function", contentType=InputTypes.StringType)
    FunctionInput.addParam("class", InputTypes.StringType, True)
    FunctionInput.addParam("type", InputTypes.StringType, True)
    VariableInput.addSub(FunctionInput, InputData.Quantity.one)
    LabelInput = InputData.parameterInputFactory("label",contentType=InputTypes.StringType)
    inputSpecification.addSub(VariableInput, InputData.Quantity.one_to_infinity)
    inputSpecification.addSub(LabelInput, InputData.Quantity.one)

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag   = 'POSTPROCESSOR DataClassifier'
    self.mapping    = {}  # dictionary for mapping input space between different DataObjects {'variableName':'externalFunctionName'}
    self.funcDict   = {}  # Contains the function to be used {'variableName':externalFunctionInstance}
    self.label      = None # ID of the variable which containf the label values
    self.outputMultipleRealizations = True # True indicate multiple realizations are returned

    # assembler objects to be requested
    self.addAssemblerObject('Function', InputData.Quantity.one_to_infinity)

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the DataClassifier post-processor.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, optional, dictionary with initialization options
      @ Out, None
    """
    super().initialize(runInfo, inputs, initDict)
    for key, val in self.mapping.items():
     self.funcDict[key] = self.retrieveObjectFromAssemblerDict('Function',val[1])

  def _handleInput(self, paramInput):
    """
      Method that handles PostProcessor parameter input block
      @ In, paramInput, ParameterInput, the already parsed input
      @ Out, None
    """
    super()._handleInput(paramInput)
    for child in paramInput.subparts:
      if child.getName() == 'variable':
        func = child.findFirst('Function')
        funcType = func.parameterValues['type']
        funcName = func.value.strip()
        self.mapping[child.parameterValues['name']] = (funcType, funcName)
      elif child.getName() == 'label':
        self.label = child.value.strip()

  def inputToInternal(self, currentInput):
    """
      Method to convert a list of input objects into the internal format that is
      understandable by this pp.
      @ In, currentInput, list, a list of DataObjects
      @ Out, newInput, list, list of converted data
    """
    if isinstance(currentInput,list) and len(currentInput) != 2:
      self.raiseAnError(IOError, "Two inputs DataObjects are required for postprocessor", self.name)
    newInput ={'classifier':{}, 'target':{}}
    haveClassifier = False
    haveTarget = False
    for inputObject in currentInput:
      if isinstance(inputObject, dict):
        newInput.append(inputObject)
      else:
        if inputObject.type not in ['PointSet', 'HistorySet']:
          self.raiseAnError(IOError, "The input for this postprocesor", self.name, "is not acceptable! Allowed inputs are 'PointSet' and 'HistorySet'.")
        if len(inputObject) == 0:
          self.raiseAnError(IOError, "The input", inputObject.name, "is empty!")
        inputDataset = inputObject.asDataset()
        inputParams = inputObject.getVars('input')
        outputParams = inputObject.getVars('output')
        dataType = None
        mappingKeys = self.mapping.keys()
        if len(set(mappingKeys)) != len(mappingKeys):
          dups = set([elem for elem in mappingKeys if mappingKeys.count(elem) > 1])
          self.raiseAnError(IOError, "The same variable {} name is used multiple times in the XML input".format(dups[0]))
        if set(self.mapping.keys()) == set(inputParams) and self.label in outputParams:
          dataType = 'classifier'
          if not haveClassifier:
            haveClassifier = True
          else:
            self.raiseAnError(IOError, "Both input data objects have been already processed! No need to execute this postprocessor", self.name)
          if inputObject.type != 'PointSet':
            self.raiseAnError(IOError, "Only PointSet is allowed as classifier, but HistorySet", inputObject.name, "is provided!")
        else:
          dataType = 'target'
          newInput[dataType]['data'] = inputObject.asDataset(outType='dict')['data']
          newInput[dataType]['dims'] = inputObject.getDimensions()
          if not haveTarget:
            haveTarget = True
          else:
            self.raiseAnError(IOError, "None of the input DataObjects can be used as the reference classifier! Either the label", \
                    self.label, "is not exist in the output of the DataObjects or the inputs of the DataObjects are not the same as", \
                    ','.join(self.mapping.keys()))
        newInput[dataType]['input'] = dict.fromkeys(inputParams)
        newInput[dataType]['output'] = dict.fromkeys(outputParams)
        if inputObject.type == 'PointSet':
          for elem in inputParams:
            newInput[dataType]['input'][elem] = copy.deepcopy(inputDataset[elem].values)
          for elem in outputParams:
            newInput[dataType]['output'][elem] = copy.deepcopy(inputDataset[elem].values)
          newInput[dataType]['type'] = inputObject.type
          newInput[dataType]['name'] = inputObject.name
        else:
          # only extract the last element in each realization for the HistorySet
          newInput[dataType]['type'] = inputObject.type
          newInput[dataType]['name'] = inputObject.name
          numRlzs = len(inputObject)
          newInput[dataType]['historySizes'] = dict.fromkeys(range(numRlzs))
          for i in range(numRlzs):
            rlz = inputObject.realization(index=i)
            for elem in inputParams:
              if newInput[dataType]['input'][elem] is None:
                newInput[dataType]['input'][elem] = np.empty(0)
              newInput[dataType]['input'][elem] = np.append(newInput[dataType]['input'][elem], rlz[elem])
            for elem in outputParams:
              if newInput[dataType]['output'][elem] is None:
                newInput[dataType]['output'][elem] = np.empty(0)
              newInput[dataType]['output'][elem] = np.append(newInput[dataType]['output'][elem], rlz[elem].values[-1])
              if newInput[dataType]['historySizes'][i] is None:
                newInput[dataType]['historySizes'][i] = len(rlz[elem].values)

    return newInput

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In,  inputIn, list, list of DataObjects
      @ Out, outputDict, dict, dictionary of outputs
    """
    inputDict = self.inputToInternal(inputIn)
    targetDict = inputDict['target']
    classifierDict = inputDict['classifier']
    outputDict = {}
    outputDict.update(inputDict['target']['data'])
    outputType = targetDict['type']
    numRlz = utils.first(targetDict['input'].values()).size
    outputDict[self.label] = []
    for i in range(numRlz):
      tempTargDict = {}
      for param, vals in targetDict['input'].items():
        tempTargDict[param] = vals[i]
      for param, vals in targetDict['output'].items():
        tempTargDict[param] = vals[i]
      tempClfList = []
      labelIndex = None
      for key, values in classifierDict['input'].items():
        calcVal = self.funcDict[key].evaluate("evaluate", tempTargDict)
        inds, = np.where(np.asarray(values) == calcVal)
        if labelIndex is None:
          labelIndex = set(inds)
        else:
          labelIndex = labelIndex & set(inds)
      if len(labelIndex) != 1:
        self.raiseAnError(IOError, "The parameters", ",".join(tempTargDict.keys()), "with values", ",".join([str(el) for el in tempTargDict.values()]), "could not be put in any class!")
      label = classifierDict['output'][self.label][list(labelIndex)[0]]
      if outputType == 'PointSet':
        outputDict[self.label].append(label)
      else:
        outputDict[self.label].append(np.asarray([label]*targetDict['historySizes'][i]))
    outputDict[self.label] = np.asarray(outputDict[self.label])
    outputDict = {'data': outputDict, 'dims':inputDict['target']['dims']}
    return outputDict

  def collectOutput(self, finishedJob, output):
    """
      Function to place all of the computed data into the output object
      @ In, finishedJob, JobHandler External or Internal instance, A JobHandler object that is in charge of running this post-processor
      @ In, output, dataObjects, The object where we want to place our computed results
      @ Out, None
    """
    super().collectOutput(finishedJob, output)
