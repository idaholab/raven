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
import importlib

from .PostProcessor import PostProcessor
from utils import InputData
from PostProcessorInterfaceBaseClass import PostProcessorInterfaceBase

class InterfacedPostProcessor(PostProcessor):
  """
    This class allows to interface a general-purpose post-processor created ad-hoc by the user.
    While the ExternalPostProcessor is designed for analysis-dependent cases, the InterfacedPostProcessor is designed more generic cases
    The InterfacedPostProcessor parses (see PostProcessorInterfaces.py) and uses only the functions contained in the raven/framework/PostProcessorFunctions folder
    The base class for the InterfacedPostProcessor that the user has to inherit to develop its own InterfacedPostProcessor is specified
    in PostProcessorInterfaceBase.py
  """

  PostProcessorInterfaces = importlib.import_module("PostProcessorInterfaces")

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
    inputSpecification = super(RavenOutput, cls).getInputSpecification()

    ## TODO: Fill this in with the appropriate tags

    return inputSpecification

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.methodToRun = None
    ## Currently, we have used both DataObject.addRealization and DataObject.load to
    ## collect the PostProcessor returned outputs. DataObject.addRealization is used to
    ## collect single realization, while DataObject.load is used to collect multiple realizations
    ## However, the DataObject.load can not be directly used to collect single realization
    self.outputMultipleRealizations = True

  def initialize(self, runInfo, inputs, initDict):
    """
      Method to initialize the Interfaced processor
      @ In, runInfo, dict, dictionary of run info (e.g. working dir, etc)
      @ In, inputs, list, list of inputs
      @ In, initDict, dict, dictionary with initialization options
      @ Out, None
    """
    PostProcessor.initialize(self, runInfo, inputs, initDict)

    inputObj = inputs[-1] if type(inputs) == list else inputs
    metaKeys = inputObj.getVars('meta')
    self.addMetaKeys(metaKeys)

  def _localReadMoreXML(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.Element, Xml element node
      @ Out, None
    """

    # paramInput = InterfacedPostProcessor.getInputSpecification()()
    # paramInput.parseNode(xmlNode)

    interfaceClasses = [c.getInputSpecification() for c in InterfacedPostProcessor.PostProcessorInterfaces.interfaceClasses()]
    paramInput = InputData.parseFromList(xmlNode, interfaceClasses)

    self.methodToRun = paramInput.getName()
    self.postProcessor = InterfacedPostProcessor.PostProcessorInterfaces.factory.returnInstance(self.methodToRun)
    if not isinstance(self.postProcessor,PostProcessorInterfaceBase):
      self.raiseAnError(IOError, 'InterfacedPostProcessor Post-Processor '+ self.name +
                        ' : not correctly coded; it must inherit the PostProcessorInterfaceBase class')

    self.postProcessor.initialize()
    self.postProcessor._handleInput(paramInput)
    if not set(self.returnFormat("input").split("|")) <= set(['HistorySet','PointSet']):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +
                        ' : self.inputFormat not correctly initialized')
    if not set(self.returnFormat("output").split("|")) <= set(['HistorySet','PointSet']):
      self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor '+ self.name +
                        ' : self.outputFormat not correctly initialized')

  def run(self, inputIn):
    """
      This method executes the interfaced  post-processor action.
      @ In, inputIn, dict, dictionary of data to process
      @ Out, outputDic, dict, dict containing the post-processed results
    """
    #FIXME THIS IS NOT CORRECT!!!!
    try:
      inputTypes = set([inp.type for inp in inputIn])
      check=True
    except AttributeError:
      check=False
    if check:
      for inp in inputIn:
        if not inputTypes <= set(self.returnFormat("input").split("|")):
          self.raiseAnError(IOError,'InterfacedPostProcessor Post-Processor named "'+ self.name +
                            '" : The input object "'+ inp.name +'" provided is of the wrong type. Got "'+
                            inp.type + '" but expected "'+self.returnFormat("input") + '"!')

    inputDic= self.inputToInternal(inputIn)
    self.raiseADebug('InterfacedPostProcessor Post-Processor '+ self.name +' : start to run')
    outputDic = self.postProcessor.run(inputDic)
    return outputDic

  def _inverse(self, inputIn):
    outputDic = self.postProcessor._inverse(inputIn)
    return outputDic

  def inputToInternal(self,inputs):
    """
      Function to convert the received input into a format this object can
      understand
      @ In, input, list, list of dataObjects handed to the post-processor
      @ Out, inputDict, list, list of dictionaries this object can process
    """
    inputDict = []
    for inp in inputs:
      if type(inp) == dict:
        return [inp]
      else:
        self.metaKeys = inp.getVars('meta')
        inputDictTemp = {}
        inputDictTemp['inpVars']   = inp.getVars('input')
        inputDictTemp['outVars']   = inp.getVars('output')
        inputDictTemp['data']      = inp.asDataset(outType='dict')['data']
        inputDictTemp['dims']      = inp.getDimensions('output')
        inputDictTemp['type']      = inp.type
        inputDictTemp['metaKeys']  = self.metaKeys
        inputDictTemp['numberRealizations'] = len(inp)
        for key in self.metaKeys:
          try:
            inputDictTemp['data'][key]  = inp.getMeta(pointwise=True,general=True)[key].values
          except:
            self.raiseADebug('The following key: ' + str(key) + ' has not passed to the Interfaced PP')
        inputDictTemp['name']     = inp.name
        inputDict.append(inputDictTemp)
    return inputDict

  def returnFormat(self,location):
    """
      Function that returns the format of either input or output
      @ In, location, str, list of dataObjects handed to the post-processor
      @ Out, form, str, format of either input or output
    """
    if location == 'input':
      form = self.postProcessor.inputFormat
    elif location == 'output':
      form = self.postProcessor.outputFormat
    return form

  def collectOutput(self, finishedJob, output, options=None):
    """
      Function to place all of the computed data into the output object, (DataObjects)
      @ In, finishedJob, object, JobHandler object that is in charge of running this PostProcessor
      @ In, output, object, the object where we want to place our computed results
      @ In, options, dict, optional, not used in PostProcessor.
        dictionary of options that can be passed in when the collect of the output is performed by another model (e.g. EnsembleModel)
      @ Out, None
    """
    PostProcessor.collectOutput(self, finishedJob, output, options=options)
