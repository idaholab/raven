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
import numpy as np
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from .PostProcessor import PostProcessor
from utils import InputData
import Files
import Runners
#Internal Modules End-----------------------------------------------------------

class ExternalPostProcessor(PostProcessor):
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
    PostProcessor.__init__(self, messageHandler)
    self.methodsToRun = []              # A list of strings specifying what
                                        # methods the user wants to compute from
                                        # the external interfaces

    self.externalInterfaces = set()     # A set of Function objects that
                                        # hopefully contain definitions for all
                                        # of the methods the user wants

    self.printTag = 'POSTPROCESSOR EXTERNAL FUNCTION'
    self.requiredAssObject = (True, (['Function'], ['n']))

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

    EFunctionInput = InputData.parameterInputFactory("Function", contentType=InputData.StringType)
    EFunctionInput.addParam("class", InputData.StringType)
    EFunctionInput.addParam("type", InputData.StringType)
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
    PostProcessor.initialize(self, runInfo, inputs, initDict)
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
    if isinstance(evaluation, Runners.Error):
      self.raiseAnError(RuntimeError, "No available output to collect (run possibly not finished yet)")

    dataLenghtHistory = {}
    inputList,outputDict = evaluation

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
                  if dataLenghtHistory[histNum+1] != len(val):
                    self.raiseAnError(IOError, key + ' the size of the arrays for history '+str(histNum+1)+' are different!')
                else:
                  dataLenghtHistory[histNum+1] = len(val)
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
      if outputDict[methodName] is None:
        self.raiseAnError(Exception,"the method "+methodName+" has not produced any result. It needs to return a result!")
      for target in input['targets']:
        if hasattr(interface, target):
          #if target not in outputDict.keys():
          if target not in methodMap.keys():
            attributeInSelf = getattr(interface, target)
            if len(np.atleast_1d(attributeInSelf)) != len(np.atleast_1d(input['targets'][target])) or (np.atleast_1d(attributeInSelf) - np.atleast_1d(input['targets'][target])).all():
              if target in outputDict.keys():
                self.raiseAWarning("In Post-Processor "+ self.name +" the modified variable "+target+
                               " has the same name of a one already modified throuhg another Function method." +
                               " This method overwrites the input DataObject variable value")
              outputDict[target] = attributeInSelf
          else:
            warningMessages.append("In Post-Processor "+ self.name +" the method "+method+
                               " has the same name of a variable contained in the input DataObject." +
                               " This method overwrites the input DataObject variable value")
    for msg in list(set(warningMessages)):
      self.raiseAWarning(msg)

    for target in input['targets'].keys():
      if target not in outputDict.keys() and target in input['targets'].keys():
        outputDict[target] = input['targets'][target]

    return outputDict
