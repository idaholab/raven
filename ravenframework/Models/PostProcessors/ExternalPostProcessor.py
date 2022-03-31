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
#External Modules---------------------------------------------------------------
import numpy as np
import copy
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessorInterface import PostProcessorInterface
from ...utils import InputData, InputTypes, utils
from ... import Files
#Internal Modules End-----------------------------------------------------------

class ExternalPostProcessor(PostProcessorInterface):
  """
    ExternalPostProcessor class. It will apply an arbitrary python function to
    a dataset and append each specified function's output to the output data
    object, thus the function should produce a scalar value per row of data. I
    have no idea what happens if the function produces multiple outputs.
  """
  def __init__(self):
    """
      Constructor
      @ In, None, dict, run info
      @ Out, None
    """
    super().__init__()
    self.methodsToRun = []              # A list of strings specifying what
                                        # methods the user wants to compute from
                                        # the external interfaces

    self.externalInterfaces = set()     # A set of Function objects that
                                        # hopefully contain definitions for all
                                        # of the methods the user wants

    self.printTag = 'POSTPROCESSOR EXTERNAL FUNCTION'
    self.addAssemblerObject('Function', InputData.Quantity.one_to_infinity)

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
    inputSpecification = super(ExternalPostProcessor, cls).getInputSpecification()

    EMethodInput = InputData.parameterInputFactory("method")
    inputSpecification.addSub(EMethodInput)

    EFunctionInput = InputData.parameterInputFactory("Function", contentType=InputTypes.StringType)
    EFunctionInput.addParam("class", InputTypes.StringType)
    EFunctionInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(EFunctionInput)

    return inputSpecification

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

    if type(currentInput) == list:
      if len(currentInput) != 1:
        self.raiseAnError(IOError, "The postprocessor ", self.name, "only allows one input DataObjects,"
              + " but multiple inputs are provided!")
      else:
        currentInput = currentInput[-1]
    assert(hasattr(currentInput, 'type')), "The type is missing for input object! We should always associate a type with it."
    inType = currentInput.type
    if inType in ['PointSet', 'HistorySet']:
      dataSet = currentInput.asDataset()
    else:
      self.raiseAnError(IOError, "Input type ", inType, ' is not yet implemented!')

    if len(currentInput) == 0:
      self.raiseAnError(IOError, 'The Input object ', currentInput.name, ' is empty!')
    inputDict = {}
    if inType == 'PointSet':
      for param in currentInput.getVars():
        inputDict[param] = copy.copy(dataSet[param].values)
    elif inType == 'HistorySet':
      sliceList = currentInput.sliceByIndex('RAVEN_sample_ID')
      indexes = currentInput.indexes
      for param in currentInput.getVars('output'):
        inputDict[param] =  [sliceData[param].dropna(indexes[-1]).values for sliceData in sliceList]
      for param in currentInput.getVars('input'):
        inputDict[param] =  [sliceData[param].values for sliceData in sliceList]
      for param in indexes:
        inputDict[param] =  [sliceData[param].dropna(indexes[-1]).values for sliceData in sliceList]

    for interface in self.externalInterfaces:
      for _ in self.methodsToRun:
        # The function should reference self and use the same variable names
        # as the xml file
        for param in interface.parameterNames():
          if param not in inputDict.keys():
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
    super().initialize(runInfo, inputs, initDict)
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
    paramInput = ExternalPostProcessor.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self._handleInput(paramInput)

  def _handleInput(self, paramInput):
    """
      Function to handle the parsed paramInput for this class.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
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
    evaluation = finishedJob.getEvaluation()
    dataLenghtHistory = {}
    inputList,outputDict = evaluation

    if isinstance(output,Files.File):
      self.raiseAWarning('Output type File not yet implemented. I am going to skip it.')
    elif output.type == 'HDF5':
      self.raiseAnError(NotImplementedError,'Output type ' + type(output).__name__
                         + ' not yet implemented!')
    elif output.type in ['PointSet', 'HistorySet']:
      output.load(outputDict, style='dict', dims=output.getDimensions())
    else:
      self.raiseAnError(IOError, 'Unknown output type: ' + str(output.type))

  def run(self, inputIn):
    """
      This method executes the postprocessor action. In this case it performs
      the action defined in the external pp
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDict, dict, Dictionary containing the post-processed results
    """
    inputDict = self.inputToInternal(inputIn)
    outputDict = {}
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
      #elif len(matchingInterfaces) == 1:
      #  methodMap[method] = (matchingInterfaces[0], method)
      else:
        for interface in matchingInterfaces:
          methodName = interface.name + '_' + method
          methodMap[methodName] = (interface, method)

    ## Evaluate the method and add it to the outputDict, also if the method
    ## adjusts the input data, then you should update it as well.
    warningMessages = []
    for methodName, (interface, method) in methodMap.items():
      # The deep copy is needed since the interface postprocesor will change the values of inputDict
      tempInputDict = copy.deepcopy(inputDict)
      outputDict[methodName] = np.atleast_1d(copy.copy(interface.evaluate(method, tempInputDict)))
      if outputDict[methodName] is None:
        self.raiseAnError(Exception,"the method "+methodName+" has not produced any result. It needs to return a result!")
      for target in tempInputDict.keys():
        if hasattr(interface, target):
          #if target not in outputDict.keys():
          if target not in methodMap.keys():
            attributeInSelf = getattr(interface, target)
            if (np.atleast_1d(attributeInSelf)).shape != (np.atleast_1d(inputDict[target])).shape or (np.atleast_1d(attributeInSelf) - np.atleast_1d(inputDict[target])).all():
              if target in outputDict.keys():
                self.raiseAWarning("In Post-Processor "+ self.name +" the modified variable "+target+
                               " has the same name of a one already modified through another Function method." +
                               " This method overwrites the input DataObject variable value")
              outputDict[target] = np.atleast_1d(attributeInSelf)
          else:
            warningMessages.append("In Post-Processor "+ self.name +" the method "+method+
                               " has the same name of a variable contained in the input DataObject." +
                               " This method overwrites the input DataObject variable value")
    for msg in list(set(warningMessages)):
      self.raiseAWarning(msg)

    # TODO: We assume the structure of input to the external pp is the same as the struture of output to this external pp
    # An interface pp should be used if the user wants to merge two data objects, or change the structures of input data
    # objects.
    numRlz = len(utils.first(outputDict.values()))
    for val in outputDict.values():
      if len(val) != numRlz:
        self.raiseAnError(IOError, "The return results from the external functions have different number of realizations!"
                + " This postpocessor ", self.name, " requests all the returned values should have the same number of realizations.")
    for target in inputDict.keys():
      if target not in outputDict.keys():
        if len(inputDict[target]) != numRlz:
          self.raiseAWarning("Parameter ", target, " is available in the provided input DataObjects,"
                  + " but it has different length from the returned values from the external functions."
                  + " Thus this parameter will not be accessible by the output DataObjects!")
        else:
          outputDict[target] = np.atleast_1d(inputDict[target])

    return outputDict
