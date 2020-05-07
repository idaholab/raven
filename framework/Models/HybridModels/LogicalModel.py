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
Created on May 6, 2020

@author: wangc
"""

#External Modules------------------------------------------------------------------------------------
import copy
import time
import numpy as np
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from .HybridModelBase import HybridModelBase
import Files
from utils import InputData, InputTypes
from utils import utils
from Runners import Error as rerror
#Internal Modules End--------------------------------------------------------------------------------

class LogicalModel(HybridModelBase):
  """
    LogicalModel Class.
    This class is aimed to automatically select the model to run among different models
    depending on the control function
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for the class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls.
    """
    inputSpecification = super(LogicalModel, cls).getInputSpecification()
    cfInput = InputData.parameterInputFactory("ControlFunction", contentType=InputTypes.StringType)
    cfInput.addParam("class", InputTypes.StringType)
    cfInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(cfInput)
    return inputSpecification

  @classmethod
  def specializeValidateDict(cls):
    """
      This method describes the types of input accepted with a certain role by the model class specialization
      @ In, None
      @ Out, None
    """
    pass

  def __init__(self,runInfoDict):
    """
      Constructor
      @ In, runInfoDict, dict, the dictionary containing the runInfo (read in the XML input file)
      @ Out, None
    """
    HybridModelBase.__init__(self,runInfoDict)
    self.printTag              = 'LogicalModel MODEL' # print tag
    self.controlFunction       = None
    # assembler objects to be requested
    self.addAssemblerObject('ControlFunction','1')

  def localInputAndChecks(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    HybridModelBase.localInputAndChecks(self, xmlNode)
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    for child in paramInput.subparts:
      if child.getName() == 'ControlFunction':
        self.controlFunction = child.value

  def initialize(self,runInfo,inputs,initDict=None):
    """
      Method to initialize this model class
      @ In, runInfo, dict, is the run info from the jobHandler
      @ In, inputs, list, is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
      @ Out, None
    """
    HybridModelBase.initialize(self,runInfo,inputs,initDict)
    self.controlFunction = self.retrieveObjectFromAssemblerDict('ControlFunction', self.controlFunction)
    if "evaluate" not in self.controlFunction.availableMethods():
      self.raiseAnError(IOError,'Function', self.controlFunction.name, 'does not contain a method named "evaluate".',
                        'It must be present if this needs to be used in a {}!'.format(self.name))
    # check models inputs and outputs

  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, newInputs, dict, dict that returns the new inputs for each sub-model
    """
    self.raiseADebug("{}: Create new input.".format(self.name))
    # TODO: standardize the way to handle code/external model/rom inputs
    modelToRun = self.controlFunction.evaluate("evaluate", kwargs)
    if modelToRun not in self.modelInstances:
      self.raiseAnError(IOError, 'Model (i.e. {}) returned from "ControlFunction" is not valid!'.format(modelToRun),
                        'Available models are: {}'.format(','.join(self.modelInstances.keys())))
    kwargs['modelToRun'] = modelToRun
    if self.modelInstances[modelToRun].type == 'Code':
      codeInput = []
      for elem in myInput:
        if isinstance(elem, Files.File):
          codeInput.append(elem)
        return (codeInput, samplerType, kwargs)
    return (myInput, samplerType, kwargs)

  def _externalRun(self,inRun, jobHandler):
    """
      Method that performs the actual run of the essembled model (separated from run method for parallelization purposes)
      @ In, inRun, tuple, tuple of Inputs (inRun[0] actual input, inRun[1] type of sampler,
        inRun[2] dictionary that contains information coming from sampler)
      @ In, jobHandler, instance, instance of jobHandler
      @ Out, exportDict, dict, dict of results from this hybrid model
    """
    self.raiseADebug("{}: External Run".format(self.name))
    originalInput = inRun[0]
    samplerType = inRun[1]
    inputKwargs = inRun[2]
    identifier = inputKwargs.pop('prefix')
    # TODO: execute control function, move this to createNewInput
    modelToRun = inputKwargs.pop('modelToRun')
    inputKwargs['prefix'] = modelToRun + utils.returnIdSeparator() + identifier
    inputKwargs['uniqueHandler'] = self.name + identifier
    moveOn = False
    while not moveOn:
      if jobHandler.availability() > 0:
        self.modelInstances[modelToRun].submit(originalInput, samplerType, jobHandler, **inputKwargs)
        self.raiseADebug("Job submitted for model", modelToRun, "with identifier", identifier)
        moveOn = True
      else:
        time.sleep(self.sleepTime)
    while not jobHandler.isThisJobFinished(inputKwargs['prefix']):
      time.sleep(self.sleepTime)
    self.raiseADebug("Job finished", modelToRun, "with identifier", identifier)
    finishedRun = jobHandler.getFinished(jobIdentifier=inputKwargs['prefix'], uniqueHandler=inputKwargs['uniqueHandler'])
    evaluation = finishedRun[0].getEvaluation()
    if isinstance(evaluation, rerror):
      self.raiseAnError(RuntimeError, "The model", modelToRun, "identified by", finishedRun[0].identifier, "failed!")
    # collect output in temporary data object
    exportDict = evaluation
    self.raiseADebug("{}: Create exportDict".format(self.name))
    return exportDict

  def collectOutput(self,finishedJob,output):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, ClientRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ Out, None
    """
    HybridModelBase.collectOutput(self, finishedJob, output)
