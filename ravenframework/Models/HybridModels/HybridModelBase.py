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
Created on April, 2020

@author: wangc
"""

#External Modules------------------------------------------------------------------------------------
import copy
import abc
import numpy as np
import itertools
from collections import OrderedDict
from ...Decorators.Parallelization import Parallel
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...Models import Dummy
from ... import Models
from ... import Files
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class HybridModelBase(Dummy):
  """
    HybridModel Base Class.
    A couple of models with the same required inputs and outputs.
    These models are assembled, and only model is executed to produce the outputs.
    The selection of models will be determined by the types of hybrid model.
  """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for the class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for specifying input of cls.
    """
    inputSpecification = super(HybridModelBase, cls).getInputSpecification()
    modelInput = InputData.parameterInputFactory("Model", contentType=InputTypes.StringType)
    modelInput.addParam("class", InputTypes.StringType)
    modelInput.addParam("type", InputTypes.StringType)
    inputSpecification.addSub(modelInput)
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
    self.modelInstances        = {}                  # dictionary {modelName: modelInstance}: instances of given model
    self.sleepTime             = 0.005               # waiting time before checking if a run is finished.
    self.printTag              = 'HybridModelBase MODEL' # print tag
    self.createWorkingDir      = False               # If the type of model is 'Code', this will set to true
    # assembler objects to be requested
    self.addAssemblerObject('Model', InputData.Quantity.one_to_infinity)

  def localInputAndChecks(self,xmlNode):
    """
      Function to read the portion of the xml input that belongs to this specialized class
      and initialize some stuff based on the inputs got
      @ In, xmlNode, xml.etree.ElementTree.Element, Xml element node
      @ Out, None
    """
    Dummy.localInputAndChecks(self, xmlNode)
    paramInput = self.getInputSpecification()()
    paramInput.parseNode(xmlNode)
    for modelNode in paramInput.findAll('Model'):
      self.modelInstances.update({modelNode.value: None})
      if not self.createWorkingDir and modelNode.parameterValues['type'] == 'Code':
        self.createWorkingDir = True

  def initialize(self,runInfo,inputs,initDict=None):
    """
      Method to initialize this model class
      @ In, runInfo, dict, is the run info from the jobHandler
      @ In, inputs, list, is a list containing whatever is passed with an input role in the step
      @ In, initDict, dict, optional, dictionary of all objects available in the step is using this model
      @ Out, None
    """
    for model in self.modelInstances:
      if isinstance(model, Models.Model):
        self.raiseAnError(IOError, "Model {} has already been initialized, and it can not be initialized again!".format(model.name))
      modelInstance = self.retrieveObjectFromAssemblerDict('Model', model)
      if modelInstance.type == 'Code':
        codeInput = []
        for elem in inputs:
          if isinstance(elem, Files.File):
            codeInput.append(elem)
        modelInstance.initialize(runInfo, codeInput, initDict)
      self.modelInstances[model] = modelInstance

  def getInitParams(self):
    """
      Method used to export to the printer in the base class the additional PERMANENT your local class have
      @ In, None
      @ Out, tempDict, dict, dictionary to be updated. {'attribute name':value}
    """
    tempDict = OrderedDict()
    return tempDict

  def getAdditionalInputEdits(self,inputInfo):
    """
      Collects additional edits for the sampler to use when creating a new input. In this case, it calls all the getAdditionalInputEdits methods
      of the sub-models
      @ In, inputInfo, dict, dictionary in which to add edits
      @ Out, None.
    """
    for _, modelInstance in self.modelInstances.items():
      modelInstance.getAdditionalInputEdits(inputInfo)

  @abc.abstractmethod
  def createNewInput(self,myInput,samplerType,**kwargs):
    """
      This function will return a new input to be submitted to the model, it is called by the sampler.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, **kwargs, dict,  is a dictionary that contains the information coming from the sampler,
           a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, newInputs, dict, dict that returns the new inputs for each sub-model
    """

  def submit(self,myInput,samplerType,jobHandler,**kwargs):
    """
     This will submit an individual sample to be evaluated by this model to a
     specified jobHandler as a client job. Note, some parameters are needed
     by createNewInput and thus descriptions are copied from there.
     @ In, myInput, list, the inputs (list) to start from to generate the new
       one
     @ In, samplerType, string, is the type of sampler that is calling to
       generate a new input
     @ In,  jobHandler, JobHandler instance, the global job handler instance
     @ In, **kwargs, dict,  is a dictionary that contains the information
       coming from the sampler, a mandatory key is the sampledVars' that
       contains a dictionary {'name variable':value}
     @ Out, None
    """
    ## Hybrid models need access to the job handler, so let's stuff it in our
    ## catch all kwargs where evaluateSample can pick it up, not great, but
    ## will suffice until we can better redesign this whole process.
    prefix = kwargs['prefix']
    kwargs['jobHandler'] = jobHandler
    jobHandler.addClientJob((self, myInput, samplerType, kwargs), self.__class__.evaluateSample, prefix, kwargs)

  @Parallel()
  def evaluateSample(self, myInput, samplerType, kwargs):
    """
      This will evaluate an individual sample on this model. Note, parameters
      are needed by createNewInput and thus descriptions are copied from there.
      @ In, myInput, list, the inputs (list) to start from to generate the new one
      @ In, samplerType, string, is the type of sampler that is calling to generate a new input
      @ In, kwargs, dict,  is a dictionary that contains the information coming from the sampler,
        a mandatory key is the sampledVars'that contains a dictionary {'name variable':value}
      @ Out, rlz, dict, This holds the output information of the evaluated sample.
    """
    self.raiseADebug("Evaluate Sample")
    kwargsKeys = list(kwargs.keys())
    kwargsKeys.pop(kwargsKeys.index("jobHandler"))
    kwargsToKeep = {keepKey: kwargs[keepKey] for keepKey in kwargsKeys}
    jobHandler = kwargs['jobHandler']
    newInput = self.createNewInput(myInput, samplerType, **kwargsToKeep)
    ## Unpack the specifics for this class, namely just the jobHandler
    result = self._externalRun(newInput,jobHandler)
    # assure rlz has all metadata
    rlz = dict((var,np.atleast_1d(kwargsToKeep[var])) for var in kwargsToKeep.keys())
    # update rlz with input space from inRun and output space from result
    rlz.update(dict((var,np.atleast_1d(kwargsToKeep['SampledVars'][var] if var in kwargs['SampledVars'] else result[var])) for var in set(itertools.chain(result.keys(),kwargsToKeep['SampledVars'].keys()))))
    return rlz

  @abc.abstractmethod
  def _externalRun(self,inRun, jobHandler):
    """
      Method that performs the actual run of the essembled model (separated from run method for parallelization purposes)
      @ In, inRun, tuple, tuple of Inputs (inRun[0] actual input, inRun[1] type of sampler,
        inRun[2] dictionary that contains information coming from sampler)
      @ In, jobHandler, instance, instance of jobHandler
      @ Out, exportDict, dict, dict of results from this hybrid model
    """

  def collectOutput(self,finishedJob,output):
    """
      Method that collects the outputs from the previous run
      @ In, finishedJob, ClientRunner object, instance of the run just finished
      @ In, output, "DataObjects" object, output where the results of the calculation needs to be stored
      @ Out, None
    """
    Dummy.collectOutput(self, finishedJob, output)

  def _extractInputs(self,dataIn, paramsList):
    """
      Extract the the parameters in the paramsList from the given data object dataIn
      @ dataIn, Instance or Dict, data object or dictionary contains the input and output parameters
      @ paramsList, List, List of parameter names
      @ localInput, numpy.array, array contains the values of selected input and output parameters
    """
    localInput = []
    if type(dataIn) == dict:
      for elem in paramsList:
        if elem in dataIn.keys():
          localInput.append(np.atleast_1d(dataIn[elem]))
        else:
          self.raiseAnError(IOError, "Parameter ", elem, " is not found!")
    else:
      self.raiseAnError(IOError, "The input type '", inputType, "' can not be accepted!")
    return np.asarray(localInput)

  def _mergeDict(self,exportDict, tempExportDict):
    """
      This function will combine two dicts into one
      @ In, exportDict, dict, dictionary stores the input, output and metadata
      @ In, tempExportDict, dict, dictionary stores the input, output and metadata
      @ Out,
    """
    if not exportDict:
      outputDict = copy.deepcopy(tempExportDict)
    else:
      outputDict = copy.deepcopy(exportDict)
      inKey = 'inputSpaceParams'
      outKey = 'outputSpaceParams'
      for key, value in tempExportDict[inKey].items():
        outputDict[inKey][key] = value
      for key, value in tempExportDict[outKey].items():
        output[outKey][key] = value
      for key, value in tempExportDict['metadata'].items():
        output['metadata'][key] = value
    self.raiseADebug("The exportDict has been updated")
    return outputDict
