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

import numpy as np

from utils import InputData, InputTypes, utils
from PluginBaseClasses.PostProcessorPluginBase import PostProcessorPluginBase

class DataClassifier(PostProcessorPluginBase):
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
        varName = child.parameterValues['name']
        if varName in self.mapping.keys():
          self.raiseAnError(IOError, "The variable {} name is duplicated in the XML input".format(varName))
        self.mapping[varName] = (funcType, funcName)
      elif child.getName() == 'label':
        self.label = child.value.strip()

  def identifyInputs(self, inputData):
    """
      Method to identify the type (i.e., 'classifier' or 'target') of input data.
      If the input data contains 'label' and required 'variables' (provided by XML input file),
      the input data will assign type 'classifier', otherwise 'target'
      Please check 'PluginsBaseClasses.PostProcessorPluginBase' for the detailed descriptions
      about 'inputData' and the output 'newInput'.
      @ In, inputData, dict, dictionary contains the input data and input files, i.e.,
          {'Data':[DataObjects.asDataset('dict')], 'Files':[FileObject]}
      @ Out, newInput, dict, dictionary of identified inputs, i.e.,
          {'classifier':DataObjects.asDataset('dict'), 'target':DataObjects.asDataset('dict')}
    """
    currentInput = inputData['Data']
    if len(currentInput) != 2:
      self.raiseAnError(IOError, "Required two inputs for PostProcessor {}, but got {}".format(self.name, len(currentInput)))
    newInput ={'classifier':{}, 'target':{}}
    haveClassifier = False
    haveTarget = False
    requiredKeys = list(self.mapping.keys()) + [self.label]
    for inputTuple in currentInput:
      _, _, inputDict = inputTuple
      if inputDict['type'] not in ['PointSet', 'HistorySet']:
        self.raiseAnError(IOError, "The input for this postprocesor", self.name, "is not acceptable! Allowed inputs are 'PointSet' and 'HistorySet'.")
      dataType = None
      if set(requiredKeys).issubset(set(inputDict['data'].keys())):
        dataType = 'classifier'
        if not haveClassifier:
          haveClassifier = True
        else:
          self.raiseAnError(IOError, "Both input data objects have been already processed! No need to execute this postprocessor", self.name)
        if inputDict['type'] != 'PointSet':
          self.raiseAnError(IOError, "Only PointSet is allowed as classifier, but got", inputDict['type'])
      else:
        dataType = 'target'
        if not haveTarget:
          haveTarget = True
        else:
          self.raiseAnError(IOError, "None of the input DataObjects can be used as the reference classifier! Either the label", \
                  self.label, "is not exist in the output of the DataObjects or the inputs of the DataObjects are not the same as", \
                  ','.join(self.mapping.keys()))
      newInput[dataType] = inputDict
    return newInput

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In,  inputIn, dict, dictionary contains the input data and input files, i.e.,
          {'Data':[DataObjects.asDataset('dict')], 'Files':[FileObject]}, only 'Data'
          will be used by this PostProcessor
      @ Out, outputDict, dict, dictionary of outputs, i.e.,
          {'data':dict of realizations, 'dim':{varName:independent dimensions that the variable depends on}}
    """
    inputDict = self.identifyInputs(inputIn)
    targetDict = inputDict['target']
    classifierDict = inputDict['classifier']
    outputDict = {}
    outputDict.update(inputDict['target']['data'])
    outputType = targetDict['type']
    dimsDict = targetDict['dims']
    numRlz = utils.first(targetDict['data'].values()).size
    outputDict[self.label] = []
    for i in range(numRlz):
      tempTargDict = {}
      for param, vals in targetDict['data'].items():
        tempTargDict[param] = vals[i]
      tempClfList = []
      labelIndex = None
      for key in self.mapping.keys():
        calcVal = self.funcDict[key].evaluate("evaluate", tempTargDict)
        values = classifierDict['data'][key]
        inds, = np.where(np.asarray(values) == calcVal)
        if labelIndex is None:
          labelIndex = set(inds)
        else:
          labelIndex = labelIndex & set(inds)
      if len(labelIndex) != 1:
        self.raiseAnError(IOError, "The parameters", ",".join(tempTargDict.keys()), "with values", ",".join([str(el) for el in tempTargDict.values()]), "could not be categorized!")
      label = classifierDict['data'][self.label][list(labelIndex)[0]]
      if outputType == 'PointSet':
        outputDict[self.label].append(label)
      else:
        historySize = 1
        for var in targetDict['data'].keys():
          dims = dimsDict[var]
          if len(dims) !=0:
            historySize = len(targetDict['data'][var][i])
            if self.label not in dimsDict:
              dimsDict[self.label] = dims
            break
        outputDict[self.label].append(np.asarray([label]*historySize))
    outputDict[self.label] = np.asarray(outputDict[self.label])
    outputDict = {'data': outputDict, 'dims':targetDict['dims']}
    return outputDict
