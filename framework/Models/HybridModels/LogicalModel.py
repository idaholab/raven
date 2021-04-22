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

  def __init__(self):
    """
      Constructor
      @ In, None
      @ Out, None
    """
    super().__init__()
    self.printTag = 'LogicalModel MODEL' # print tag
    # Function object that is used to control the execution of models
    self.controlFunction = None
    # assembler objects to be requested
    self.addAssemblerObject('ControlFunction', InputData.Quantity.one)

  def localInputAndChecks(self, xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    HybridModelBase.localInputAndChecks(self, xmlNode)
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    self.controlFunction = paramInput.findFirst('ControlFunction').value
    if self.controlFunction is None:
      self.raiseAnError(IOError, '"ControlFunction" is required for "{}", but it is not provided!'.format(self.name))

  def initialize(self, runInfo, inputs, initDict=None):
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
    # check models inputs and outputs, we require all models under LogicalModel should have
    # exactly the same inputs and outputs from RAVEN piont of view.
    # TODO: currently, the above statement could not fully verified by the following checks.
    # This is mainly because: 1) we do not provide prior check for Codes, 2) we do not have
    # a standardize treatment of variables in ExternalModels and ROMs
    # if DataObjects among the inputs, check input consistency with ExternalModels and ROMs
    inpVars = None
    inpObjName = None
    for inpObj in inputs:
      if inpObj.type in ['PointSet', 'HistorySet']:
        if not inpVars:
          inputVars = inpObj.getVars('input')
          inpObjName = inpObj.name
        else:
          self.raiseAnError(IOError, 'Only one input DataObject can be accepted for {}!'.format(self.name))
      elif inpObj.type in ['DataSet']:
        self.raiseAnError(IOError, 'DataSet "{}" is not allowed as input for {}!'.format(inpObj.name, self.name))

    extModelVars = None
    extModelName = None
    romInpVars = None
    romOutVars = None
    romName = None
    for modelName, modelInst in self.modelInstances.items():
      if modelInst.type == 'ExternalModel':
        vars = list(modelInst.modelVariableType.keys())
        if not extModelVars:
          extModelVars = vars
          extModelName = modelName
        elif set(extModelVars) != set(vars):
          self.raiseAnError(IOError,'"Variables" provided to model "{}" are not the same as model "{}"!'.format(modelName, extModelName))
      elif modelInst.type == 'ROM':
        inpVars = modelInst.initializationOptionDict['Features']
        outVars = modelInst.initializationOptionDict['Target']
        if not romInpVars:
          romInpVars = inpVars
          romOutVars = outVars
          romName = modelName
        elif set(romInpVars) != set(inpVars) or set(romOutVars) != set(outVars):
          self.raiseAnError(IOError,'ROM "{}" does not have the same Features and Targets as ROM "{}"!'.format(modelName, romName))
      elif modelInst.type == 'Code':
        self.raiseAWarning('The input/output consistency check is not performed for Model "{}" among {}!'.format(modelName, self.name))
      else:
        self.raiseAnError(IOError, 'Model "{}" with type "{}" can not be accepted by {}!'.format(modelName, modelInst.type, self.name))
    if extModelVars is not None and romInpVars is not None:
      romVars = romInpVars + romOutVars
      if set(romVars) != set(extModelVars):
        self.raiseAnError(IOError,'"Variables" provided to model "{}" are not the same as ROM "{}"!'.format(extModelName, romName),
                          'The variables listed in "Target" and "Features" of ROM should be also listed for ExternalModel under "Variables" node!')
    if extModelVars is not None or romInpVars is not None:
      if not inputVars:
        modelName = romName if romInpVars else extModelName
        self.raiseAnError(IOError, 'An input DataObject is required for {}!'.format(self.name),
                          'This is because model "{}" expects a DataObject as input.'.format(modelName))
      else:
        error = set(inputVars) - set(romInpVars) if romInpVars else set(inputVars) - set(extModelVars)
        modelName = romName if romInpVars else extModelName
        if len(error) > 0:
          self.raiseAnError(IOError, 'Variable(s) listed under DataObject "{}" could not be find in model "{}"!'.format(','.join(error), modelName))

  def createNewInput(self, myInput, samplerType, **kwargs):
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
          codeInput.append(copy.deepcopy(elem))
      return (codeInput, samplerType, kwargs)
    return (myInput, samplerType, kwargs)

  def _externalRun(self, inRun, jobHandler):
    """
      Method that performs the actual run of the logical model (separated from run method for parallelization purposes)
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

  def collectOutput(self, finishedJob, output):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, ClientRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ Out, None
    """
    HybridModelBase.collectOutput(self, finishedJob, output)
