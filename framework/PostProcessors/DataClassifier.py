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

#External Modules---------------------------------------------------------------
import copy
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from BaseClasses import BaseType
from utils import InputData
from .PostProcessor import PostProcessor
import MessageHandler
#Internal Modules End-----------------------------------------------------------

class DataClassifier(PostProcessor):
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
    inputSpecification = super(DataClassifier, cls).getInputSpecification()
    VariableInput = InputData.parameterInputFactory("variable", contentType=InputData.StringType)
    VariableInput.addParam("name", InputData.StringType, True)
    FunctionInput = InputData.parameterInputFactory("Function", contentType=InputData.StringType)
    FunctionInput.addParam("class", InputData.StringType, True)
    FunctionInput.addParam("type", InputData.StringType, True)
    VariableInput.addSub(FunctionInput, InputData.Quantity.one)
    LabelInput = InputData.parameterInputFactory("label",contentType=InputData.StringType)
    inputSpecification.addSub(VariableInput, InputData.Quantity.one_to_infinity)
    inputSpecification.addSub(LabelInput, InputData.Quantity.one)

    return inputSpecification

  def __init__(self, messageHandler):
    """
      Constructor
      @ In, messageHandler, MessageHandler, message handler object
      @ Out, None
    """
    PostProcessor.__init__(self, messageHandler)
    self.printTag   = 'POSTPROCESSOR DataClassifier'
    self.dynamic    = False
    self.mapping    = {}  # dictionary for mapping input space between different DataObjects {'variableName':'externalFunctionName'}
    self.funcDict   = {}  # Contains the function to be used {'variableName':externalFunctionInstance}
    self.label      = None

  def _localGenerateAssembler(self, initDict):
    """
      It is used for sending to the instanciated class, which is implementing the method, the objects that have
      been requested throught "whatDoINeed" method
      @ In, initDict, dict, dictionary ({'mainClassName:{specializedObjectName:ObjectInstance}'})
      @ Out, None
    """
    availableFunc = initDict['Functions']
    for key, val in self.mapping.items():
      if val[1] not in availableFunc.keys():
        self.raiseAnError(IOError, 'Function ', val[1], ' was not found among the available functions: ', availableFunc.keys())
      self.funcDict[key] = availableFunc[val[1]]
      # check if the correct method is present
      if 'evaluate' not in self.funcDict[key].availableMethods():
        self.raiseAnError(IOError, 'Function ', val[1], ' does not contain a method named "evaluate". It mush be present if this needs to be used by other RAVEN entities!')

  def _localWhatDoINeed(self):
    """
      This method is a local mirror of the general whatDoINeed method that need to request
      special objects, e.g. Functions
      @ In, None
      @ Out, needDict, list, list of objects needed
    """
    needDict = {}
    needDict['Functions'] = []
    for func in self.mapping.values():
      needDict['Functions'].append(func)
    return needDict

  def initialize(self, runInfo, inputs, initDict=None):
    """
      Method to initialize the pp.
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)

  def _localReadMoreXML(self, xmlNode):
    """
      Method to read the portion of the XML input that belongs to this specialized class
      @ In, xmlNode, xml.etree.Element, XML element node
      @ Out, None
    """
    paramInput = DataClassifier.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Method that handles PostProcessor parameter input block
      @ In, paramInput, ParameterInput, the already parsed input
      @ Out, None
    """
    for child in paramInput.subparts:
      if child.getName() == 'variable':
        func = child.findFirst('Function')
        funcType = func.parameterValues['type']
        funcName = func.value.strip()
        self.mapping[child.getName()] = (funcType, funcName)
      elif child.getName() == 'label':
        self.label = child.value.strip()

  def inputToInternal(self, currentInput):
    """
      Method to convert a list of input objects into the internal format that is
      understandable by this pp.
      @ In, currentInput, list, a list of DataObjects
      @ Out, newInput, list, list of converted data
    """
    if type(currentInput) != list and len(currentInput) != 2:
      self.raiseAnError(IOError, "Two inputs DataObjects are required for postprocessor", self.name)
    newInput ={'classifier':{}, 'target':{}}
    haveClassifier = False
    haveTarget = False
    for inputObject in currentInput:
      if type(inputObject) == dict:
        newInput.append(inputObject)
      else:
        if not hasattr(inputObject, 'type') and inputObject.type not in ['PointSet', 'HistorySet']:
          self.raiseAnError(IOError, "The input for this postprocesor", self.name, "is not acceptable! Allowed inputs are 'PointSet' and 'HistorySet'.")
        if inputObject.isItEmpty():
          self.raiseAnError(IOError, "The input", inputObject.name, "is empty!")
        inputParams = inputObject.getParaKeys('inputs')
        outputParams = inputObject.getParaKeys('outputs')
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
            self.raiseAnError(IOError, "Both input data objects have been already classifier! No need to execute this postprocessor", self.name)
          if inputObject.type != 'PointSet':
            self.raiseAnError(IOError, "Only PointSet is allowed as classifier, but HistorySet", inputObject.name, "is provided!")
        else:
          dataType = 'target'
          if not haveTarget:
            haveTarget = True
          else:
            self.raiseAnError(IOError, "None of the input DataObjects can be used as the reference classifier! Either the label", \
                    self.label, "is not exist in the output of the DataObjects or the inputs of the DataObjects are not the same as", \
                    ','.join(self.mapping.keys()))

        newInput[dataType]['input'] = copy.deepcopy(inputObject.getInpParametersValues())
        newInput[dataType]['output'] = copy.deepcopy(inputObject.getOutParametersValues())
        newInput[dataType]['type'] = inputObject.type
        newInput[dataType]['name'] = inputObject.name

    return newInput

  def run(self, inputIn):
    """
      This method executes the postprocessor action.
      @ In,  inputIn, list, list of DataObjects
      @ Out, None
    """
    inputDict = self.inputToInternal(inputIn)
    targetDict = inputDict['target']
    classifierDict = inputDict['classifier']
    outputDict = {}
    outputType = targetDict['type']
    outputDict['dataType'] = outputType
    outputDict['dataFrom'] = targetDict[name]

    if outputType == 'PointSet':
      outputDict[self.label] = np.empty(0)
      numRlz = targetDict['input'].values()[0].size
      for i in range(numRlz):
        tempTargDict = {}
        for param, vals in targetDict['input'].items():
          tempTargDict[param] = vals[i]
        tempClfList = []
        labelIndex = None
        for key, values in classifierDict['input'].items():
          calcVal = self.funcDict[key].evaluate("evaluate", tempTargDict)
          inds, = np.where(values == calcVal)
          if labelIndex is None:
             lableIndex = set(inds)
          else:
            labelIndex = labelIndex & set(inds)
        if len(labelIndex) != 1:
          self.raiseAnError(IOError, "The parameters", ",".join(tempTargDict.keys()), "with values", ",".join([str(el) for el in tempTargDict.values()]), "could not be classifier!")
        outputDict[self.label] = np.append(outputDict[self.label], classifierDict['output'][self.label][labelIndex[0]])

    else: # HistorySet
      pass

    return outputDict

  def collectOutput(self, finishedJob, output):
    """
      Method to place all of the computed data into output object
      @ In, finishedJob, object, JobHandler object that is in charge of running this postprocessor
      @ In, output, object, the object where we want to place our computed results
      @ Out, None
    """
    evaluation = finishedJob.getEvaluation()
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "Job ", finishedJob.identifier, " failed!")
    inputObject, outputDict = evaluation

    if isinstance(output, Files.File):
      self.raiseAnError(IOError, "Dump results to files is not yet implemented!")

    for inp in inputObject:
      if inp.name == outputDict['dataFrom']:
        inputObject = inputObject[0]
        break
    if inputObject != output:
      ## Copy any data you need from the input DataObject before adding new data
      if output.type == 'PointSet':
        for key in output.getParaKeys('inputs') + output.getParaKeys('outputs'):
          col = None
          if key in inputObject.getParaKeys('inputs'):
            col = copy.copy(inputObject.getParam('inputs', key))
          elif key in inputObject.getParaKeys('outputs'):
            col = copy.copy(inputObject.getParam('outputs', key))
          if col is not None:
            for val in col:
              if key in output.getParaKeys('inputs'):
                output.updateInputValue(key, value)
              else:
                output.updateOutputValue(key,value)
      elif output.type == 'HistorySet':
        for key in output.getParaKeys('inputs') + output.getParaKeys('outputs'):
          col = None
          if key in inputObject.getParaKeys('inputs'):
            col = {}
            for histIdx in inputObject.getParametersValues('inputs').keys():
              col[histIdx] = copy.copy(inputObject.getParam('inputs', histIdx)[key])
          elif key in inputObject.getParaKeys('outputs'):
            col = {}
            for histIdx in inputObject.getParametersValues('outputs').keys():
              col[histIdx] = copy.copy(inputObject.getParam('outputs', histIdx)[key])
          if col is not None:
            for histIdx, vals in col.items():
              if key in output.getParaKeys('inputs'):
                output.updateInputValue([histIdx, key], vals[-1])
              else:
                output.updateOutputValue([histIdx, key], vals)

    if output.type == 'PointSet':
      for val in outputDict[self.label]:
        output.updateOutputValue(self.label, val)
    elif output.type == 'HistorySet':
      histKey = output.getOutParametersValues.keys()
      for ind, key in enumerate(histKey):
        output.updateOutputValue([key, self.label], outputDict[self.label][ind,:])

